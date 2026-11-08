"""PyTorch (cuBLAS) SGEMM baseline on the local GPU: prints median ms and GFLOP/s per size."""
import sys
import torch

def bench(n, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
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
    gflops = 2 * n**3 / (med * 1e-3) / 1e9
    print(f"torch.mm n={n:5d}  median {med:8.3f} ms  {gflops:8.1f} GFLOP/s")

if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    for n in [int(x) for x in sys.argv[1:]] or [256, 512, 1024, 2048]:
        bench(n)
