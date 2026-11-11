"""Remeasure gemm-bench's saved native timers, interleaved across implementations."""

import argparse
import ctypes
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--previous-artifacts", type=Path, required=True)
    parser.add_argument("--openblas", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bench_repo))
    from mimir.benchmark import Kernel as MimKernel
    from pytorch.benchmark import Kernel as TorchKernel
    from common.openblas import OpenBLASKernel

    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.results_dir
    old = args.previous_artifacts
    default = root / "default/mimir/artifacts"
    aligned = root / "aligned/mimir/artifacts"
    results = json.loads((root / "default/mimir/results.json").read_text())
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    ptlib = ctypes.CDLL(
        str(root / "default/pytorch/artifacts/1024_1024_1024/pytorch_timer.so")
    )
    ptlib.create_kernel.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ptr,
        ptr,
        ptr,
    ]
    ptlib.create_kernel.restype = ctypes.c_void_p
    ptlib.measure.argtypes = [ctypes.c_void_p, ctypes.c_int]
    ptlib.measure.restype = ctypes.c_double
    ptlib.destroy_kernel.argtypes = [ctypes.c_void_p]
    ptlib.destroy_kernel.restype = None
    ptlib.last_error.restype = ctypes.c_char_p
    oblib = ctypes.CDLL(str(args.openblas))
    oblib.openblas_set_num_threads(1)
    obtimerlib = root / "default/openblas/openblas_timer.so"
    obtimer = ctypes.CDLL(str(obtimerlib)).measure
    obtimer.argtypes = [ctypes.c_int] * 3 + [ptr] * 3 + [ctypes.c_int]
    obtimer.restype = ctypes.c_double
    out = []
    identities = []
    for record in results["rows"]:
        if "error" in record:
            raise RuntimeError(record)
        shape = record["shape"]
        m, n, k = shape
        stem = f"{m}_{n}_{k}"
        a = np.load(old / stem / "A.npy")
        b = np.load(old / stem / "B.npy")
        ref = a.astype("float64") @ b.astype("float64")
        case = dict(op="gemm", name=record["name"], shape=shape, reps=record["reps"])
        kernels = {}
        for name, p in [
            ("old", old),
            ("mc", default),
            ("aligned", aligned),
            ("current", root / "current-default/mimir/artifacts"),
            ("rows96", root / "row-regs/mimir/artifacts"),
        ]:
            so = p / stem / "kernel.so"
            kernels[name] = MimKernel(so, a, b, (m, n), so.parent, 0, "clang++", [], [])
            identities.append(
                dict(
                    case=record["name"],
                    variant=name,
                    path=str(so),
                    sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
                )
            )
        kernels["openblas"] = OpenBLASKernel(obtimer, case, a, b, obtimerlib)
        kernels["pytorch"] = TorchKernel(
            ptlib, case, a, b, 0, root / "default/pytorch", {}
        )
        errors = {}
        for name, kernel in kernels.items():
            value = kernel.run()
            np.testing.assert_allclose(value, ref, rtol=0.001, atol=0.001)
            errors[name] = float(np.max(abs(value - ref)))
            for _ in range(5):
                kernel.run()
        timings = {name: [] for name in kernels}
        counts = {name: [] for name in kernels}
        orders = []
        rng = random.Random(42)
        for batch in range(9):
            order = list(kernels)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                kernel = kernels[name]
                timings[name].append(kernel.measure(100, case["reps"]) * 1000)
                counts[name].append(kernel.last_iterations)
        row = dict(
            name=record["name"],
            shape=shape,
            ms={name: statistics.median(v) for name, v in timings.items()},
            batches_ms=timings,
            counts=counts,
            orders=orders,
            max_abs_error=errors,
        )
        out.append(row)
        print(json.dumps(dict(name=row["name"], ms=row["ms"])), flush=True)
        (root / "investigation-paired.json").write_text(
            json.dumps(
                dict(
                    affinity=sorted(os.sched_getaffinity(0)),
                    threads=1,
                    warmup=5,
                    repeats=9,
                    min_repeat_ms=100,
                    order_seed=42,
                    identities=identities,
                    rows=out,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
