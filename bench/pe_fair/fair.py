import os, sys, ctypes, time, statistics, json
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {5})
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.manual_seed(0)

lib = ctypes.CDLL(sys.argv[1])
libc = ctypes.CDLL("libc.so.6")
libc.free.argtypes = [ctypes.c_void_p]

def ours(name, out_shape, *inputs):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_void_p] * len(inputs)
    ptrs = [x.data_ptr() for x in inputs]
    n = 1
    for s in out_shape: n *= s
    def run():
        p = fn(*ptrs)
        libc.free(p)
    def value():
        p = fn(*ptrs)
        buf = (ctypes.c_float * n).from_address(p)
        t = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(out_shape).clone()
        libc.free(p)
        return t
    return run, value

def timeit(f, reps=30, warm=5):
    for _ in range(warm): f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); f(); ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts) / 1e6, min(ts) / 1e6

x  = torch.randn(2048, 2048)
p  = torch.randn(8, 64, 112, 112)
pc = p.contiguous(memory_format=torch.channels_last)
q, k, v = torch.randn(2048, 64), torch.randn(2048, 64), torch.randn(2048, 64)

cases = {
  "transpose":  (lambda: x.t().contiguous(), ("transpose_", (2048, 2048), x)),
  "row_sum":    (lambda: x.sum(1), ("row_sum_", (2048,), x)),
  "col_sum":    (lambda: x.sum(0), ("col_sum_", (2048,), x)),
  "softmax":    (lambda: torch.softmax(x, 1), ("softmax_", (2048, 2048), x)),
  "layer_norm": (lambda: F.layer_norm(x, (2048,)), ("layer_norm_", (2048, 2048), x)),
  "max_pool":   (lambda: F.max_pool2d(p, 2, 2), ("max_pool_", (8, 64, 56, 56), p)),
  "avg_pool":   (lambda: F.avg_pool2d(p, 3, 1), ("avg_pool_", (8, 64, 110, 110), p)),
  "attention":  (lambda: torch.softmax((q @ k.t()) * 0.125, 1) @ v, ("attention_", (2048, 64), q, k, v)),
}
extra = {
  "max_pool": lambda: F.max_pool2d(pc, 2, 2),
  "avg_pool": lambda: F.avg_pool2d(pc, 3, 1),
  "attention": lambda: F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], scale=0.125),
}

rows = []
for name, (eager, spec) in cases.items():
    run, value = ours(spec[0], spec[1], *spec[2:])
    ref = eager()
    got = value()
    err = ((got - ref).abs() / (ref.abs() + 1e-3)).max().item()
    comp = torch.compile(eager, dynamic=False)
    comp()
    r = {"case": name, "relerr": err,
         "ours": timeit(run), "eager": timeit(eager), "compile": timeit(comp)}
    if name in extra: r["eager_alt"] = timeit(extra[name])
    rows.append(r)
    print(json.dumps(r), flush=True)
