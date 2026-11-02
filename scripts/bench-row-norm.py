#!/usr/bin/env python3
"""Build the grouped row normalization benchmark; run the printed binary after checking build activity."""

import pathlib
import subprocess
import sys


root = pathlib.Path(__file__).resolve().parent.parent
directory = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "build/task14-row-bench"
subprocess.run(
    [sys.executable, str(root / "lit/tensor/row_norm.py"), str(root / "build/bin/mim"),
     str(root / "lit/tensor/row_norm.mim"), str(directory), "benchmark"],
    check=True,
)
subprocess.run(
    ["/opt/homebrew/opt/llvm/bin/clang++", "-O3", "-mcpu=native", "-Wno-override-module",
     "-I", str(directory), str(root / "lit/tensor/row_norm_bench.cpp"), str(directory / "probe.ll"),
     "-o", str(directory / "bench")],
    check=True,
)
print(directory / "bench")
