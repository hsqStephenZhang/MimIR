#!/usr/bin/env python3
"""Build the grouped row normalization benchmark; run the printed binary after checking build activity."""

import argparse
import pathlib
import subprocess
import sys


root = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=pathlib.Path, default=root / "build/task15-row-bench")
parser.add_argument("--rows", type=int, default=1)
args = parser.parse_args()
if args.rows <= 0:
    parser.error("--rows must be positive")
directory = args.directory.resolve()
subprocess.run(
    [sys.executable, str(root / "lit/tensor/row_norm.py"), str(root / "build/bin/mim"),
     str(root / "lit/tensor/row_norm.mim"), str(directory), "benchmark", f"rows={args.rows}"],
    check=True,
)
subprocess.run(
    ["/opt/homebrew/opt/llvm/bin/clang++", "-O3", "-mcpu=native", "-Wno-override-module",
     "-I", str(directory), str(root / "lit/tensor/row_norm_bench.cpp"), str(directory / "probe.ll"),
     "-o", str(directory / "bench")],
    check=True,
)
print(directory / "bench")
