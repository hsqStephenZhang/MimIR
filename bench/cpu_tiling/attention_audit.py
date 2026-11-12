"""Untimed worker audit for CPU attention's default and online schedules."""

import ctypes
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from attention_cpu import HERE, SOURCE, MASKS, environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("threads", type=int, nargs="?", choices=[4, 6, 8])
    parser.add_argument("--root", type=Path, default=Path("/tmp/mimir-attention-local"))
    parser.add_argument("--variants", nargs="+", choices=["mimir_default", "mimir_online", "mimir_joint"],
                        default=["mimir_default", "mimir_online"])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.threads is None:
        runtime = Path(json.loads((root / "metadata.json").read_text())["omp_runtime"])
        for threads in [4, 6, 8]:
            for variant in args.variants:
                path = root / f"t{threads}" / variant / "h8_s128_d64"
                command = ["clang++", "-O3", "-march=haswell", "-mno-avx512f", "-shared", "-fPIC",
                           "-std=c++20", "-Wno-override-module", str(path / "kernel.ll"),
                           str(HERE / "attention_native.cpp"), str(HERE / "audit_omp.cpp"),
                           str(root / "omp-audit.o"), str(runtime), "-Wl,-rpath," + str(runtime.parent),
                           "-o", str(path / "audit.so")]
                subprocess.run(command, check=True)
            subprocess.run([sys.executable, __file__, str(threads), "--root", str(root),
                            "--variants", *args.variants],
                           env=environment(threads), check=True)
        return
    threads = args.threads
    os.sched_setaffinity(0, MASKS[threads])
    import numpy as np
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    path = root / "inputs/h8_s128_d64"
    q, k, v = [np.load(path / (name + ".npy")) for name in ["q", "k", "v"]]
    out = np.empty_like(q)
    os.sched_setaffinity(0, {0})
    records = []
    for variant in args.variants:
        lib = ctypes.CDLL(str(root / f"t{threads}" / variant / "h8_s128_d64/audit.so"))
        lib.run_attention.argtypes = [ptr] * 4 + [ctypes.c_size_t]
        assert lib.run_attention(q, k, v, out, out.size) == 0
        assert np.isfinite(out).all()
        lib.audit_workers.restype = lib.audit_cpus.restype = ctypes.c_uint64
        workers, cpus = lib.audit_workers(), lib.audit_cpus()
        record = dict(variant=variant, launches=lib.audit_launches(), team=lib.audit_team(),
                      workers=[i for i in range(64) if workers >> i & 1],
                      cpus=[i for i in range(64) if cpus >> i & 1])
        expected = json.loads((root / f"t{threads}" / variant / "h8_s128_d64/record.json").read_text())["launches"]
        assert record["launches"] == expected
        if expected:
            assert record["team"] == threads
            assert record["workers"] == list(range(threads)) and record["cpus"] == MASKS[threads]
        else:
            assert record["launches"] == 0 and record["team"] == 0
        records.append(record)
    (root / f"t{threads}/worker-audit.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(threads=threads, records=records)), flush=True)


if __name__ == "__main__":
    main()
