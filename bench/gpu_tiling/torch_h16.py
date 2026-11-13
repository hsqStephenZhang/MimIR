"""PyTorch (cuBLAS) 16-bit GEMM baseline: prints median ms and TFLOP/s per size and dtype."""
import sys
import torch

def bench(n, dtype, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.mm(a, b)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    med = times[len(times) // 2]
    print(f"torch.mm {str(dtype):15s} n={n:5d}  median {med:8.4f} ms  {2 * n**3 / (med * 1e-3) / 1e12:6.1f} TFLOP/s")

if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    for n in [int(x) for x in sys.argv[1:]] or [1024, 2048, 4096, 8192]:
        for dtype in (torch.float16, torch.bfloat16):
            bench(n, dtype)
