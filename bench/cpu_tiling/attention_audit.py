"""Untimed worker audit for CPU attention's default and online schedules."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

from attention_cpu import HERE, SOURCE, MASKS, environment

ROOT = Path("/tmp/mimir-attention-local")


def main():
    if len(sys.argv) == 1:
        runtime = Path(json.loads((ROOT / "metadata.json").read_text())["omp_runtime"])
        for threads in [4, 6, 8]:
            for variant in ["mimir_default", "mimir_online"]:
                path = ROOT / f"t{threads}" / variant / "h8_s128_d64"
                command = ["clang++", "-O3", "-march=haswell", "-mno-avx512f", "-shared", "-fPIC",
                           "-std=c++20", "-Wno-override-module", str(path / "kernel.ll"),
                           str(HERE / "attention_native.cpp"), str(HERE / "audit_omp.cpp"),
                           str(ROOT / "omp-audit.o"), str(runtime), "-Wl,-rpath," + str(runtime.parent),
                           "-o", str(path / "audit.so")]
                subprocess.run(command, check=True)
            subprocess.run([sys.executable, __file__, str(threads)], env=environment(threads), check=True)
        return
    threads = int(sys.argv[1])
    os.sched_setaffinity(0, MASKS[threads])
    import numpy as np
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    path = ROOT / "inputs/h8_s128_d64"
    q, k, v = [np.load(path / (name + ".npy")) for name in ["q", "k", "v"]]
    out = np.empty_like(q)
    os.sched_setaffinity(0, {0})
    records = []
    for variant in ["mimir_default", "mimir_online"]:
        lib = ctypes.CDLL(str(ROOT / f"t{threads}" / variant / "h8_s128_d64/audit.so"))
        lib.run_attention.argtypes = [ptr] * 4 + [ctypes.c_size_t]
        assert lib.run_attention(q, k, v, out, out.size) == 0
        assert np.isfinite(out).all()
        lib.audit_workers.restype = lib.audit_cpus.restype = ctypes.c_uint64
        workers, cpus = lib.audit_workers(), lib.audit_cpus()
        record = dict(variant=variant, launches=lib.audit_launches(), team=lib.audit_team(),
                      workers=[i for i in range(64) if workers >> i & 1],
                      cpus=[i for i in range(64) if cpus >> i & 1])
        if variant == "mimir_default":
            assert record["launches"] == 2 and record["team"] == threads
            assert record["workers"] == list(range(threads)) and record["cpus"] == MASKS[threads]
        else:
            assert record["launches"] == 0 and record["team"] == 0
        records.append(record)
    (ROOT / f"t{threads}/worker-audit.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(threads=threads, records=records)), flush=True)


if __name__ == "__main__":
    main()
