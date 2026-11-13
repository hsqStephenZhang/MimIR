"""Compiles a MimIR kernel template for a tritonbench operator, checks it against PyTorch and times it with nsys.

usage: tb.py <op> key=value...   e.g. tb.py softmax M=4096 N=4096 R=4 I=32
Run it with tritonbench's venv (torch); the kernel time is the sum of the program's kernels per call.
"""
import csv
import math
import os
import subprocess
import sys

import torch

ROOT  = os.path.dirname(os.path.abspath(__file__))
MIMIR = os.path.abspath(os.path.join(ROOT, "..", ".."))
OUT   = os.path.join(ROOT, "out")
ENV   = dict(os.environ)
HOME  = os.path.expanduser("~")
ENV["PATH"] = ":".join([f"{HOME}/opt/llvm-23/bin", f"{HOME}/opt/tools/bin", f"{HOME}/opt/cuda/bin",
                        f"{HOME}/opt/src/llvm-build/bin", ENV["PATH"]])


def gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


# op: (template, inputs(p) -> tensors, reference(*inputs) -> tensor, tolerance)
OPS = {
    "softmax": ("softmax.mim",
                lambda p: [torch.randn(p["M"], p["N"], dtype=torch.float16)],
                lambda x: torch.softmax(x.float(), -1).half(), 2e-3),
    "rms_norm": ("rms_norm.mim",
                 lambda p: [torch.randn(p["M"], p["N"], dtype=torch.float16), torch.rand(p["N"], dtype=torch.float16)],
                 lambda x, w: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half(),
                 2e-2),
    "layer_norm": ("layer_norm.mim",
                   lambda p: [torch.randn(p["M"], p["N"], dtype=torch.float16),
                              torch.rand(p["N"], dtype=torch.float16), torch.rand(p["N"], dtype=torch.float16)],
                   lambda x, w, b: torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(),
                                                                  1e-5).half(), 2e-2),
    "addmm": ("addmm.mim",
              lambda p: [torch.randn(p["M"], p["K"], dtype=torch.float16) / 8,
                         torch.randn(p["K"], p["N"], dtype=torch.float16) / 8, torch.randn(p["N"], dtype=torch.float16)],
              lambda a, b, bias: (a.float() @ b.float() + bias.float()).half(), 2e-2),
    "linear_gelu": ("addmm.mim",
                    lambda p: [torch.randn(p["M"], p["K"], dtype=torch.float16) / 8,
                               torch.randn(p["K"], p["N"], dtype=torch.float16) / 8,
                               torch.randn(p["N"], dtype=torch.float16)],
                    lambda a, b, bias: gelu(a.float() @ b.float() + bias.float()).half(), 2e-2),
    "swiglu": ("swiglu.mim",
               lambda p: [torch.randn(p["M"], p["N"], dtype=torch.float16),
                          torch.randn(p["M"], p["N"], dtype=torch.float16)],
               lambda a, b: (torch.nn.functional.silu(a.float()) * b.float()).half(), 2e-2),
}


def render(op, params):
    text = open(os.path.join(ROOT, OPS[op][0])).read()
    levels = {
        "ws": "serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red_ws ((0, 0, 64), 1, 3, @S@, 0, wg_mma (64, @TN@, 16))))",
        "pws": "serial_persistent (132, serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red_ws ((0, 0, 64), 1, 3, @S@, @PP@, wg_mma (64, @TN@, 16)))))",
        "tma": "serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red_tma ((0, 0, 64), 1, 3, @S@, wg_mma (64, @TN@, 16))))",
    }
    text = text.replace("@LEVELS@", levels.get(params.get("L", ""), "@LEVELS@"))
    text = text.replace("@OP@", op)
    text = text.replace("@ACT@", {"linear_gelu": "gelu", "addmm": ""}.get(op, ""))
    for key, value in params.items():
        text = text.replace(f"@{key}@", str(value))
    if "N" in params:
        text = text.replace("@INV_N@", repr(1.0 / params["N"]))
    for key, value in params.items():
        text = text.replace(f"@{key}@", str(value))
    return text


def kernel_ms(rep):
    out = subprocess.run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                          rep], capture_output=True, text=True, env=ENV).stdout
    rows = [r for r in csv.reader(out.splitlines()) if len(r) > 8 and r[0][:1].isdigit()]
    return sum(float(r[3]) for r in rows) / 1e6, len(rows)


def main():
    op     = sys.argv[1]
    params = dict(arg.split("=", 1) for arg in sys.argv[2:] if "=" in arg)
    params = {k: int(v) if v.isdigit() else v for k, v in params.items()}
    flags  = [arg for arg in sys.argv[2:] if "=" not in arg]
    name   = op + "_" + "_".join(f"{k}{v}" for k, v in params.items())
    os.makedirs(OUT, exist_ok=True)
    os.chdir(OUT)
    open(f"{name}.mim", "w").write(render(op, params))
    mim = os.environ.get("MIM", os.path.join(MIMIR, "build-h200", "bin", "mim"))
    log = subprocess.run([mim, "-p", "opt", "-p", "gpu", "-X", "tensor:tile-fuse=1", f"{name}.mim", "-p", "nvptx",
                          "-X", "nvptx:sm=90a", *flags], capture_output=True, text=True, env=ENV)
    open(f"{name}.log", "w").write(log.stdout + log.stderr)
    if log.returncode or "unrealized" in log.stderr:
        print(f"{name}: mim failed\n" + "\n".join(l for l in (log.stdout + log.stderr).splitlines()
                                                  if "error" in l or "unrealized" in l)[:2000])
        return 1
    fn = f"tb_{op}"
    subprocess.run(["clang++", "-O2", f"-DFN={fn}", os.path.join(ROOT, "tb_driver.cpp"), f"{name}.ll", "-lcuda", "-o",
                    f"{name}.exec", "-Wno-override-module"], check=True, env=ENV)
    torch.manual_seed(0)
    inputs = OPS[op][1](params)
    paths  = []
    for i, t in enumerate(inputs):
        path = f"{name}.in{i}"
        t.contiguous().numpy().tofile(path)
        paths.append(path)
    ref  = OPS[op][2](*inputs)
    size = ref.numel() * ref.element_size()
    subprocess.run([f"./{name}.exec", "1", f"{name}.out", str(size), *paths], check=True, env=ENV)
    got = torch.from_file(f"{name}.out", size=ref.numel(), dtype=ref.dtype).reshape(ref.shape)
    err = ((got.float() - ref.float()).abs() / (1 + ref.float().abs())).max().item()
    ok  = err <= OPS[op][3] and not math.isnan(err)
    subprocess.run(["nsys", "profile", "-o", name, "--force-overwrite", "true", f"./{name}.exec", "5", f"{name}.out",
                    str(size), *paths], capture_output=True, env=ENV)
    ms, kernels = kernel_ms(f"{name}.nsys-rep")
    print(f"{name}: {'OK' if ok else 'FAIL'} (err {err:.2e}) {ms:.4f} ms/call, {kernels} kernel(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
