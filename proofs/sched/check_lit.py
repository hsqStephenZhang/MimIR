#!/usr/bin/env python3
"""Runs `sched_check` on the schedule trees of every lit test that loads `tensor`.

Usage: check_lit.py <repo> <mim> [<sched_check>]
Each test is compiled once per set of `-X` options its `RUN` lines pass to `%mim -p opt`.
"""
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

repo, mim = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
check = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), ".lake/build/bin/sched_check")

jobs = []
for pattern in ("lit/**/*.mim", "extra/*/lit/**/*.mim"):
    for path in sorted(glob.glob(os.path.join(repo, pattern), recursive=True)):
        text = open(path, errors="replace").read()
        if "tensor" not in text:
            continue
        options = set()
        for line in text.splitlines():
            if line.startswith("// RUN:") and "%mim" in line and "-p opt" in line:
                options.add(tuple(x for x in re.findall(r"-X (\S+)", line) if not x.startswith("ll:") and "%" not in x))
        jobs += [(path, xs) for xs in sorted(options)]


def compile_one(job):
    path, xs = job
    cmd = [mim, "-VVVV", "-p", "opt"] + [a for x in xs for a in ("-X", x)] + [path, "-o", os.devnull]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.path.dirname(path))
        return run.stderr + run.stdout
    except subprocess.TimeoutExpired:
        return ""


with ThreadPoolExecutor(8) as pool:
    log = "".join(pool.map(compile_one, jobs))
result = subprocess.run([check], input=log, capture_output=True, text=True)
lines = result.stdout.splitlines()
print("\n".join(line for line in lines if not line.startswith("ok:")))
print(f"{len(jobs)} compilations")
sys.exit(result.returncode)
