import torch, time
dev = "cuda"
def bench(f, iters=20):
    f(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3
x = torch.randn(12, 1024, 1024, device=dev)
print(f"softmax (12,1024,1024): {bench(lambda: torch.softmax(x, -1)):.3f} ms")
q = torch.randn(12, 512, 64, device=dev); k = torch.randn_like(q); v = torch.randn_like(q)
def manual():
    s = (q @ k.transpose(1, 2)) * 0.125
    return torch.softmax(s, -1) @ v
print(f"attention L=512 manual (2 matmul + softmax): {bench(manual):.3f} ms")
print(f"attention L=512 F.sdpa: {bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)):.3f} ms")
