"""PyTorch attention baselines on the local GPU: median ms and TFLOP/s of F.scaled_dot_product_attention per shape."""
import itertools
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


VARIANTS = [
    ("fp16 flash", torch.float16, SDPBackend.FLASH_ATTENTION, False),
    ("bf16 flash", torch.bfloat16, SDPBackend.FLASH_ATTENTION, False),
    ("fp32 efficient", torch.float32, SDPBackend.EFFICIENT_ATTENTION, False),
    ("tf32 efficient", torch.float32, SDPBackend.EFFICIENT_ATTENTION, True),
    ("fp32 math", torch.float32, SDPBackend.MATH, False),
    ("tf32 math", torch.float32, SDPBackend.MATH, True),
]


def bench(bh, l, d, causal):
    flops = 4 * bh * l * l * d / (2 if causal else 1)
    for name, dtype, backend, tf32 in VARIANTS:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        q, k, v = (torch.randn(1, bh, l, d, device="cuda", dtype=dtype) for _ in range(3))
        try:
            with sdpa_kernel(backend):
                ms = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
            print(f"BH={bh:4d} L={l:5d} D={d:4d} causal={int(causal)} {name:15s} {ms:9.4f} ms {flops / ms / 1e9:8.1f} TFLOP/s")
        except RuntimeError as error:
            print(f"BH={bh:4d} L={l:5d} D={d:4d} causal={int(causal)} {name:15s} unsupported ({str(error).splitlines()[0][:60]})")
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.__version__)
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    shapes = [(32, 1024, 64), (32, 2048, 64)] if quick else itertools.product([32, 128], [1024, 2048, 4096, 8192], [64, 128])
    for bh, l, d in shapes:
        for causal in (False, True):
            if l * l * bh * 4 > 16 << 30:
                continue
            bench(bh, l, d, causal)
