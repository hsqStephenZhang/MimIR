"""PyTorch 16-bit attention baselines (FA2 flash backend, cuDNN backend): median ms and TFLOP/s per shape, D = 64, no mask."""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch_attn import timed

if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.__version__)
    shapes = [(32, 1024), (32, 2048), (32, 4096), (128, 2048), (128, 4096)]
    for bh, l in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            q, k, v = (torch.randn(1, bh, l, 64, device="cuda", dtype=dtype) for _ in range(3))
            for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
                try:
                    with sdpa_kernel(backend):
                        ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
                    print(f"BH={bh:4d} L={l:5d} {str(dtype):15s} {name:6s} {ms:8.4f} ms {4 * bh * l * l * 64 / ms / 1e9:7.1f} TFLOP/s")
                except RuntimeError as error:
                    print(f"BH={bh:4d} L={l:5d} {str(dtype):15s} {name:6s} unsupported ({str(error).splitlines()[0][:60]})")
