"""Compile single-kernel fusion programs to PTX, then time them against PyTorch on any CUDA host.

prepare: python ptx_bench.py prepare --mim build/bin/mim --sm 86 --out DIR rowsum softmax_resident ...
run:     python ptx_bench.py run DIR [--repeats 9] [--min-ms 100]
"""

import argparse
import ctypes as C
import itertools
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE = (12, 1024, 1024)


def specs(torch):
    def softmax(x):
        return torch.softmax(x, -1)

    def rowsum(x):
        return x.sum(-1)

    def attention(q, k, v):
        q, k, v = (x.reshape(2, 128, 64) for x in (q, k, v))
        return torch.softmax(q @ k.transpose(1, 2) * 0.125, -1) @ v

    return {
        "attn_warps": ([(256, 64)] * 3, (2, 128, 64), attention),
        "rowsum": ([SHAPE], (12, 1024), rowsum),
        "softmax_resident": ([SHAPE], SHAPE, softmax),
        "softmax_resident_shared": ([SHAPE], SHAPE, softmax),
        "softmax_fused_online": ([SHAPE], SHAPE, softmax),
        "softmax_resident8": ([SHAPE], SHAPE, softmax),
    }


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in args.programs:
        work = out / name
        work.mkdir(exist_ok=True)
        (work / "kernel.mim").write_text((HERE / f"{name}.mim").read_text())
        mim = [str(args.mim.resolve()), "-p", "opt", "-p", "gpu", "-X", "tensor:tile-fuse=1", "kernel.mim",
               "-p", "nvptx", "-X", "nvptx:no-embed", "-X", f"nvptx:sm={args.sm}"]
        with (work / "compile.log").open("w") as log:
            subprocess.run(mim, cwd=work, check=True, stdout=log, stderr=subprocess.STDOUT)
            cmds = [mim]
            dev = "kernel_dev.ll"
            if "@__nv_" in (work / dev).read_text():
                cmds.append(["llvm-link", "--only-needed", dev, str(args.libdevice), "-o", "kernel_raw.bc"])
                dev = "kernel_raw.bc"
            cmds += [
                ["opt", "-mtriple=nvptx64-nvidia-cuda", "-passes=nvvm-reflect,default<O3>", "-unroll-threshold=600",
                 dev, "-o", "kernel.bc"],
                ["llc", "-march=nvptx64", f"-mcpu=sm_{args.sm}", "kernel.bc", "-o", "kernel.ptx"],
            ]
            for cmd in cmds[1:]:
                subprocess.run(cmd, cwd=work, check=True, stdout=log, stderr=subprocess.STDOUT)
        host = (work / "kernel.ll").read_text()
        kernels = set(re.findall(r'c"(tileKernel_\d+|kernel_\d+)', host))
        launches = re.findall(
            r"cuLaunchKernel_ptsz\(ptr %[^,]+, i32 (\d+), i32 1, i32 1, i32 (\d+), i32 1, i32 1, i32 (\d+)", host)
        if len(kernels) != 1 or len(launches) != 1:
            raise RuntimeError(f"{name}: expected one kernel with a static launch")
        groups, items, smem = map(int, launches[0])
        info = dict(program=name, sm=args.sm, kernel=kernels.pop(), groups=groups, items=items, smem=smem,
                    commands=cmds)
        (work / "launch.json").write_text(json.dumps(info, indent=2))
        print(name, info["kernel"], groups, items, smem, flush=True)


class Driver:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.lib.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p),
                                                                           C.c_void_p]

    def check(self, r, what):
        if r:
            raise RuntimeError(f"{what}: CUDA driver error {r}")


def run(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    drv = Driver()
    table = specs(torch)
    rows = []
    for work in sorted(p.parent for p in args.bundle.glob("*/launch.json")):
        info = json.loads((work / "launch.json").read_text())
        name = info["program"]
        in_shapes, out_shape, ref = table[name]
        gen = torch.Generator(device=dev).manual_seed(42)
        inputs = [torch.randn(s, device=dev, generator=gen) for s in in_shapes]
        expect = ref(*inputs)
        mod, fn = C.c_void_p(), C.c_void_p()
        drv.check(drv.lib.cuModuleLoadData(C.byref(mod), (work / "kernel.ptx").read_bytes() + b"\0"), "load")
        drv.check(drv.lib.cuModuleGetFunction(C.byref(fn), mod, info["kernel"].encode()), "function")
        out = torch.empty(out_shape, device=dev)
        bufs = inputs + [out]
        # A program may pass the same input several times; extra pointer slots alias the first input.
        param = re.search(r"_param_0\[(\d+)\]", (work / "kernel.ptx").read_text())
        bufs = inputs * (int(param.group(1)) // 8 - len(bufs)) + bufs
        stream = torch.cuda.current_stream().cuda_stream
        order = None
        for perm in itertools.permutations(range(len(bufs))):
            ptrs = (C.c_ulonglong * len(bufs))(*[bufs[i].data_ptr() for i in perm])
            params = (C.c_void_p * 1)(C.cast(ptrs, C.c_void_p))
            out.fill_(float("nan"))
            drv.check(drv.lib.cuLaunchKernel(fn, info["groups"], 1, 1, info["items"], 1, 1, info["smem"], stream,
                                             params, None), "launch")
            torch.cuda.synchronize()
            if torch.allclose(out, expect, rtol=1e-3, atol=1e-3 if name.startswith("attn") else 1e-5):
                order = perm
                break
        if order is None:
            print(f"{name}: FAILED validation (max err {float((out - expect).abs().nan_to_num(1e30).max()):.3g})")
            rows.append(dict(program=name, error="validation"))
            continue
        err = float((out.double() - ref(*[x.double() for x in inputs])).abs().max())

        def mimir():
            drv.lib.cuLaunchKernel(fn, info["groups"], 1, 1, info["items"], 1, 1, info["smem"], stream, params, None)

        def baseline():
            ref(*inputs)

        def batch(f):
            f()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            f()
            end.record()
            end.synchronize()
            n = max(10, int(args.min_ms / max(start.elapsed_time(end), 1e-3)))
            start.record()
            for _ in range(n):
                f()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / n

        times = {"mimir": [], "pytorch": []}
        for _ in range(args.repeats):
            times["mimir"].append(batch(mimir))
            times["pytorch"].append(batch(baseline))
        m, t = statistics.median(times["mimir"]), statistics.median(times["pytorch"])
        nbytes = sum(x.numel() * 4 for x in inputs) + out.numel() * 4
        row = dict(program=name, mimir_ms=m, pytorch_ms=t, speedup=t / m, min_traffic_gb_s=nbytes / m / 1e6,
                   max_abs_err_vs_fp64=err, launch=dict(groups=info["groups"], items=info["items"],
                                                        smem=info["smem"]), abi_order=order,
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        rows.append(row)
        print(f"{name:26s} mimir {m:.4f} ms  pytorch {t:.4f} ms  speedup {t / m:.3f}  "
              f"({nbytes / m / 1e6:.0f} GB/s at minimal traffic)", flush=True)
    (args.bundle / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--mim", type=Path, required=True)
    pp.add_argument("--sm", type=int, required=True)
    pp.add_argument("--out", type=Path, required=True)
    pp.add_argument("--libdevice", type=Path, default=Path("/opt/cuda/nvvm/libdevice/libdevice.10.bc"))
    pp.add_argument("programs", nargs="+")
    pr = sub.add_parser("run")
    pr.add_argument("bundle", type=Path)
    pr.add_argument("--repeats", type=int, default=9)
    pr.add_argument("--min-ms", type=float, default=100)
    args = p.parse_args()
    prepare(args) if args.cmd == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
