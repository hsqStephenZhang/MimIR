"""One PyTorch operator run 30 times, for a kernel-time profile: `nsys profile python torch_kern.py <what> ...`.
`gemm <n> <f16|bf16>`, `softmax`, `rowsum`, `attn <L> <fp32|f16>` (two matmuls and a softmax), `sdpa <L> <fp32|f16>`."""
import sys, torch
what = sys.argv[1]
if what == "gemm":
    n = int(sys.argv[2]); dt = {"f16": torch.float16, "bf16": torch.bfloat16}[sys.argv[3]]
    a = torch.randn(n, n, device="cuda", dtype=dt); b = torch.randn(n, n, device="cuda", dtype=dt); f = lambda: torch.mm(a, b)
elif what == "softmax":
    x = torch.randn(12, 1024, 1024, device="cuda"); f = lambda: torch.softmax(x, -1)
elif what == "rowsum":
    x = torch.randn(12, 1024, 1024, device="cuda"); f = lambda: x.sum(-1)
else:
    l = int(sys.argv[2]); dt = torch.float16 if sys.argv[3] == "f16" else torch.float32
    q = torch.randn(12, l, 64, device="cuda", dtype=dt); k = torch.randn_like(q); v = torch.randn_like(q)
    if what == "attn": f = lambda: torch.softmax((q @ k.transpose(1, 2)) * 0.125, -1) @ v
    else: f = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
for _ in range(30): f()
torch.cuda.synchronize()
