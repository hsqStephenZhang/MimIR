"""PyTorch baselines for C = A · (W ∘ s) with A F32, W I8 and a per-column F32 scale: prints median ms per variant."""
import sys
import torch


def timed(fn, iters=200):
    for _ in range(20):
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


def bench(n):
    a = torch.randn(n, n, device="cuda")
    w = torch.randint(-127, 128, (n, n), device="cuda", dtype=torch.int8)
    s = torch.rand(n, device="cuda") * 0.02
    wf = w.float() * s
    compiled = torch.compile(lambda a, w, s: a @ (w.float() * s))
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        tag = "tf32" if tf32 else "fp32"
        rows = {
            "dequant+mm (eager)": lambda: a @ (w.float() * s),
            "dequant+mm (torch.compile)": lambda: compiled(a, w, s),
            "mm on pre-dequantized W": lambda: a @ wf,
        }
        for name, fn in rows.items():
            ms = timed(fn)
            print(f"n={n} {tag} {name:28s} median {ms:7.4f} ms  {2 * n**3 / (ms * 1e-3) / 1e9:8.0f} GFLOP/s")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.__version__)
    for n in [int(x) for x in sys.argv[1:]] or [1024]:
        bench(n)
