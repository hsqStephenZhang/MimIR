import os, sys, ctypes, time, statistics, json
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {5})
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.manual_seed(0)

libc = ctypes.CDLL("libc.so.6")
libc.free.argtypes = [ctypes.c_void_p]
libs = {kv.split("=", 1)[0]: ctypes.CDLL(kv.split("=", 1)[1]) for kv in sys.argv[1:]}

class Out(ctypes.Structure):
    pass

def ours(lib, name, outs, *inputs):
    fn = getattr(lib, name)
    ptrs = [x.data_ptr() for x in inputs]
    if len(outs) == 1:
        fn.restype = ctypes.c_void_p
    else:
        class R(ctypes.Structure):
            _fields_ = [(f"p{i}", ctypes.c_void_p) for i in range(len(outs))]
        fn.restype = R
    fn.argtypes = [ctypes.c_void_p] * len(inputs)
    def ps(r):
        return [r] if len(outs) == 1 else [getattr(r, f"p{i}") for i in range(len(outs))]
    def run():
        for p in ps(fn(*ptrs)): libc.free(p)
    def value():
        res = []
        for p, shape in zip(ps(fn(*ptrs)), outs):
            n = 1
            for s in shape: n *= s
            buf = (ctypes.c_float * n).from_address(p)
            res.append(torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(shape).clone())
            libc.free(p)
        return res
    return run, value

def timeit(f, reps=40, warm=5):
    for _ in range(warm): f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); f(); ts.append(time.perf_counter_ns() - t0)
    return round(statistics.median(ts) / 1e6, 3)

a, b, bias = torch.randn(1024, 512), torch.randn(512, 1024), torch.randn(1024, 1024)
pa, pr = torch.randn(1021, 509), torch.randn(2045, 2045)
b2, d2 = torch.randn(512, 512), torch.randn(512, 1024)
a2, b3 = torch.randn(512, 1024), torch.randn(1024, 512)
bm, table = torch.randn(512, 1024), torch.randn(8192, 512)
gidx = torch.randint(0, 8192, (1024, 512))
gidx16 = gidx.to(torch.int16).contiguous()
q, k, v = torch.randn(2048, 64), torch.randn(2048, 64), torch.randn(2048, 64)

def ln(x):
    return F.layer_norm(x, (1024,))

cases = {
    "mm_stats":  (lambda: ((p := a @ b).sum(1), p.amax(0)), [(1024,), (1024,)], (a, b)),
    "mm_two":    (lambda: (torch.relu(p := a @ b), p + bias), [(1024, 1024), (1024, 1024)], (a, b, bias)),
    "mm_rowsum": (lambda: ((p := a @ b), p.sum(1)), [(1024, 1024), (1024,)], (a, b)),
    "mm_ln":     (lambda: ln(a @ b), [(1024, 1024)], (a, b)),
    "k64_rowsum": (lambda: ((p := q @ k.t()), p.sum(1)), [(2048, 2048), (2048,)], (q, k.t().contiguous())),
    "k64_rowmax": (lambda: ((p := q @ k.t()), p.amax(1)), [(2048, 2048), (2048,)], (q, k.t().contiguous())),
    "attention": (lambda: torch.softmax((q @ k.t()) * 0.125, 1) @ v, [(2048, 64)], (q, k, v)),
    "pad_mm":     (lambda: F.pad(pa, (1, 2, 1, 2)) @ b, [(1024, 1024)], (pa, b)),
    "pad_relu":   (lambda: torch.relu(F.pad(pr, (1, 2, 1, 2))), [(2048, 2048)], (pr,)),
    "relu_chain": (lambda: torch.relu(a @ b2) @ d2, [(1024, 1024)], (a, b2, d2)),
    "plain_chain": (lambda: (a @ b2) @ d2, [(1024, 1024)], (a, b2, d2)),
    "relu_mm":    (lambda: torch.relu(a) @ bm, [(1024, 1024)], (a, bm)),
    "gather_mm":  (lambda: torch.gather(table, 0, gidx) @ bm, [(1024, 1024)], (table, gidx16, bm)),
    "perm_chain": (lambda: (a2 @ b3).t() @ d2, [(512, 1024)], (a2, b3, d2)),
}

for name, (eager, outs, inputs) in cases.items():
    ref = eager()
    ref = list(ref) if isinstance(ref, tuple) else [ref]
    row = {"case": name}
    for lname, lib in libs.items():
        run, value = ours(lib, name, outs, *inputs)
        err = max(((g - r).abs() / (r.abs() + 1e-2)).max().item() for g, r in zip(value(), ref))
        row[lname] = timeit(run)
        row[lname + "_err"] = float(f"{err:.1e}")
    row["eager"] = timeit(eager)
    comp = torch.compile(eager, dynamic=False)
    comp()
    row["compile"] = timeit(comp)
    print(json.dumps(row), flush=True)
