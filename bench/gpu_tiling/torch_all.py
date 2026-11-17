"""PyTorch baselines on the current GPU: median of CUDA-event timings, in ms."""
import torch, torch.nn.functional as F

def med(f, iters=100, warm=10):
    for _ in range(warm): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]

print(torch.cuda.get_device_name(0), torch.__version__)
for n in (1024, 2048, 4096, 8192):
    row = [f"gemm n={n}"]
    for name, dt, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True), ("f16", torch.float16, False), ("bf16", torch.bfloat16, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a = torch.randn(n, n, device="cuda", dtype=dt); b = torch.randn(n, n, device="cuda", dtype=dt)
        t = med(lambda: torch.mm(a, b), iters=50 if n >= 4096 else 100)
        row.append(f"{name} {t:.4f} ({2 * n**3 / (t * 1e-3) / 1e12:.0f} TF)")
    print(" | ".join(row))
torch.backends.cuda.matmul.allow_tf32 = False
x = torch.randn(12, 1024, 1024, device="cuda")
print(f"softmax (12,1024,1024) fp32: {med(lambda: torch.softmax(x, -1)):.4f}")
print(f"rowsum  (12,1024,1024) fp32: {med(lambda: x.sum(-1)):.4f}")
for (h, l, d) in ((12, 512, 64), (12, 1024, 64), (12, 2048, 64), (12, 4096, 64), (12, 1024, 128)):
    for name, dt in (("fp32", torch.float32), ("f16", torch.float16)):
        q = torch.randn(h, l, d, device="cuda", dtype=dt); k = torch.randn_like(q); v = torch.randn_like(q)
        def manual():
            s = (q @ k.transpose(1, 2)) * (d ** -0.5)
            return torch.softmax(s, -1) @ v
        print(f"attention H={h} L={l} D={d} {name}: manual {med(manual):.4f} | F.sdpa {med(lambda: F.scaled_dot_product_attention(q, k, v)):.4f}")
