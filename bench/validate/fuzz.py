"""Checks compiled tile schedules against the untiled program, on random shapes and schedules.

usage: fuzz.py <op> [cases] [seed]   e.g. fuzz.py addmm 20
Each case renders a `bench/tritonbench` template, compiles it twice and runs both on the same inputs:
the reference with every `tile_req (op, …)` replaced by `op`, its meaning, compiled for the CPU, and the schedule under
test with `tensor:tile-fuse=1` for the GPU.
Inputs are small integers, so every sum and product of a contraction is exact in the input and accumulator types and
any reassociation must agree bit for bit; ops with transcendental functions compare within a tolerance instead.
A schedule the tile rules reject at elaboration counts as `rule`, one the lowering rejects later as `late`; neither is a
failure, but a `late` one marks a limit the rules do not state yet.
"""
import os
import random
import re
import subprocess
import sys

import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
TB    = os.path.join(HERE, "..", "tritonbench")
MIMIR = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT   = os.path.join(HERE, "out")
HOME  = os.path.expanduser("~")
ENV   = dict(os.environ)
ENV["PATH"] = ":".join([f"{HOME}/opt/llvm-23/bin", f"{HOME}/opt/tools/bin", f"{HOME}/opt/cuda/bin",
                        f"{HOME}/opt/src/llvm-build/bin", ENV["PATH"]])
EXTRA = os.environ.get("MIM_FLAGS", "").split()
MIM = os.environ.get("MIM", os.path.join(MIMIR, "build-h200", "bin", "mim"))

sys.path.insert(0, TB)
from tb import render  # noqa: E402


def render_local(op, params):
    text = open(os.path.join(HERE, f"{op}.mim")).read()
    text = text.replace("@LEVELS@", LEVELS.get(params.get("L", ""), "@LEVELS@"))
    for key, value in params.items():
        text = text.replace(f"@{key}@", str(value))
    return text


def pick(choices):
    return random.choice(choices)


# op: (sample params, inputs(p) -> arrays, output (shape, dtype), exact)
def addmm_params():
    tm, tn = pick([64, 128]), pick([64, 128, 256])
    p = {"M": pick([64, 128, 192, 256, 320]), "N": pick([64, 128, 192, 256]), "K": pick([64, 128, 192, 256]),
         "TM": tm, "TN": tn, "TAIL": pick([0, 1]), "S": pick([2, 3, 4]), "L": pick(["red", "tma", "ws", "pws"]),
         "PP": pick([0, 1])}
    return p


def small_ints(shape, lo=-2, hi=2):
    return np.random.randint(lo, hi + 1, size=shape)


from tb import render as _tb_render  # noqa: E402

LEVELS = {
    "red": "serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red ((0, 0, 64), 1, 3, wg_mma (64, @TN@, 16))))",
    "tma": "serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red_tma ((0, 0, 64), 1, 3, @S@, wg_mma (64, @TN@, 16))))",
    "ws": "serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red_ws ((0, 0, 64), 1, 3, @S@, @PP@, wg_mma (64, @TN@, 16))))",
    "pws": "serial_persistent (132, serial_group ((@TM@, @TN@, 0), @TAIL@, 0, group_warpgroup ((64, @TN@, 0), 1, wg_red_ws ((0, 0, 64), 1, 3, @S@, @PP@, wg_mma (64, @TN@, 16)))))",
}

OPS = {
    "addmm": (addmm_params,
              lambda p: [small_ints((p["M"], p["K"])).astype(np.float16), small_ints((p["K"], p["N"])).astype(np.float16),
                         small_ints((p["N"],)).astype(np.float16)],
              lambda p: ((p["M"], p["N"]), np.float16), True),
    "gemm": (lambda: dict(addmm_params(), L=pick(["red", "tma", "ws", "pws"])),
             lambda p: [small_ints((p["M"], p["K"])).astype(np.float16), small_ints((p["K"], p["N"])).astype(np.float16)],
             lambda p: ((p["M"], p["N"]), np.float32), True),
    "softmax": (lambda: {"M": pick([4, 8, 64, 100]), "N": pick([32, 64, 256, 1000, 1024]), "R": pick([1, 2, 4]),
                         "I": pick([4, 8, 16, 32]), "P": pick([0, 1])},
                lambda p: [np.random.randn(p["M"], p["N"]).astype(np.float16)],
                lambda p: ((p["M"], p["N"]), np.float16), False),
    "rms_norm": (lambda: {"M": pick([4, 8, 64]), "N": pick([64, 256, 1024]), "R": pick([1, 2, 4]),
                          "I": pick([4, 8, 16])},
                 lambda p: [np.random.randn(p["M"], p["N"]).astype(np.float16),
                            np.random.rand(p["N"]).astype(np.float16)],
                 lambda p: ((p["M"], p["N"]), np.float16), False),
}


def first_error(log):
    line = next((l for l in log.splitlines() if "error" in l or "unrealized" in l), log[-300:])
    return line.split("error: ")[-1].split(": ", 1)[-1][:160] if "unrealized" in line else line.split("error: ")[-1][:160]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV, **kw)


def build(name, text, flags):
    open(f"{name}.mim", "w").write(text)
    log = run([MIM, *flags[0], f"{name}.mim", *flags[1]])
    if log.returncode or "unrealized" in log.stderr and "tile-fuse=1" in " ".join(flags[0]):
        return None, (log.stdout + log.stderr)
    exe = run(["clang++", "-O1", f"-DFN=tb_{OPNAME}", os.path.join(TB, "tb_driver.cpp"), f"{name}.ll", "-lcuda", "-o",
               f"{name}.exec", "-Wno-override-module"])
    if exe.returncode:
        return None, exe.stderr
    return f"./{name}.exec", ""


def main():
    global OPNAME
    OPNAME = sys.argv[1]
    cases  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seed   = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    random.seed(seed)
    np.random.seed(seed)
    sample, inputs, output, exact = OPS[OPNAME]
    os.makedirs(OUT, exist_ok=True)
    os.chdir(OUT)
    tally = {"pass": 0, "fail": 0, "rule": 0, "late": 0, "error": 0}
    for case in range(cases):
        p    = sample()
        name = f"{OPNAME}_{case}"
        text = render_local(OPNAME, p) if os.path.exists(os.path.join(HERE, f"{OPNAME}.mim")) else render(OPNAME, p)
        spec = re.sub(r"tile_req \((\w+), [^;]*\)\);", r"\1);", text)
        ref, why = build(f"{name}_ref", spec, (["-p", "opt"], ["-p", "ll", "-X", f"ll:o={name}_ref.ll"]))
        if not ref:
            kind = "rule" if f"{name}_ref.mim:" in why else "error"
            tally[kind] += 1
            print(f"{name} {p}: {kind}: {first_error(why)}")
            continue
        test, why = build(name, text, (["-p", "opt", "-p", "gpu", "-X", "tensor:tile-fuse=1", *EXTRA],
                                       ["-p", "nvptx", "-X", "nvptx:sm=90a"]))
        if not test:
            kind = "rule" if f"{name}.mim:" in why else "late"
            tally[kind] += 1
            print(f"{name} {p}: {kind}: {first_error(why)}")
            continue
        args  = []
        for i, a in enumerate(inputs(p)):
            a.tofile(f"{name}.in{i}")
            args.append(f"{name}.in{i}")
        shape, dtype = output(p)
        size  = int(np.prod(shape)) * np.dtype(dtype).itemsize
        outs  = []
        for exe, tag in ((ref, "ref"), (test, "test")):
            r = run([exe, "1", f"{name}.{tag}.out", str(size), *args], timeout=600)
            if r.returncode:
                outs = None
                break
            outs.append(np.fromfile(f"{name}.{tag}.out", dtype=dtype).reshape(shape))
        if outs is None:
            tally["error"] += 1
            print(f"{name} {p}: a run failed")
            continue
        want, got = outs
        if exact:
            bad = np.argwhere(want.view(np.uint16) != got.view(np.uint16)) if dtype == np.float16 else \
                  np.argwhere(want != got)
        else:
            w, g = want.astype(np.float64), got.astype(np.float64)
            bad = np.argwhere(~(np.abs(w - g) <= 2e-2 * (1 + np.abs(w))))
        if len(bad):
            tally["fail"] += 1
            i = tuple(bad[0])
            print(f"{name} {p}: FAIL at {len(bad)} elements, first {i}: want {want[i]} got {got[i]}")
        else:
            tally["pass"] += 1
            print(f"{name} {p}: ok")
    print(tally)
    return 1 if tally["fail"] or tally["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
