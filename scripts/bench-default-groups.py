#!/usr/bin/env python3
"""Build marker-free default-pipeline workloads; time only when no compiler build is active."""

import argparse
import os
import pathlib
import subprocess
import time


root = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=pathlib.Path, default=root / "build/task17-default-bench")
parser.add_argument("--build-only", action="store_true")
args = parser.parse_args()
directory = args.directory.resolve()
directory.mkdir(parents=True, exist_ok=True)
template = (root / "lit/tensor/group_default.mim").read_text().split("extern fun gemm")[0]
header, cases = [], []
workloads = []
for size in (256, 512, 1024, 2048):
    for kind in ("gemm", "mlp"):
        workloads.append((kind, size, size))
for m, n in ((1024, 1024), (4096, 1024), (1024, 4096)):
    for kind in ("soft", "layer"):
        workloads.append((kind, m, n))
workloads.append(("conv", 32, 8))
for group in (0, 1):
    text = template
    for index, (kind, m, n) in enumerate(workloads):
        name = f"g{group}_{index}"
        if kind in ("gemm", "mlp"):
            third = f"«{n}; F32»" if kind == "gemm" else f"«{n}, {n}; F32»"
            signature = f"a: «{m}, {n}; F32», b: «{n}, {n}; F32», c: {third}"
            expression = f"{'gemm_relu' if kind == 'gemm' else 'mlp_chain'} (buf a, buf b, buf c)"
            shape = f"({m}, {n})"
        elif kind in ("soft", "layer"):
            signature = f"a: «{m}, {n}; F32», b c: F32"
            expression = f"{'row_soft' if kind == 'soft' else 'row_layer'} (buf a)"
            shape = f"({m}, {n})"
        else:
            signature = f"a: «2, {n}, {m}, {m}; F32», b c: «{n}, {n}, 3, 3; F32»"
            expression = (
                f"conv Ring @(2, {n}, {n}, {m - 2}, {m - 2}, 3, 3) ((1, 1), (1, 1), (0, 0)) "
                f"(conv Ring @(2, {n}, {n}, {m}, {m}, 3, 3) ((1, 1), (1, 1), (0, 0)) "
                "(buf a, buf b), buf c)"
            )
            shape = f"(2, {n}, {m - 4}, {m - 4})"
        text += f"extern fun {name} ({signature}): «{shape}; F32» = return ({expression});\n"
        if kind in ("soft", "layer"):
            header.append(f'extern "C" float* {name}(float*, float, float);')
        else:
            header.append(f'extern "C" float* {name}(float*, float*, float*);')
    source = directory / f"group{group}.mim"
    source.write_text(text)
    with (directory / f"group{group}.log").open("w") as log:
        subprocess.run(
            [str(root / "build/bin/mim"), "-p", "opt", "-p", "ll", "-X", f"tensor:group={group}",
             "-X", f"ll:o={directory / f'group{group}.ll'}", str(source)],
            stdout=log, stderr=log, check=True,
        )
for index, (kind, m, n) in enumerate(workloads):
    functions = [f"g{group}_{index}" for group in (1, 0)]
    if kind in ("soft", "layer"):
        functions = [f"[](float* a, float*, float*) {{ return {fn}(a, 0, 0); }}" for fn in functions]
    cases.append(f'{{"{kind}", {m}, {n}, ' + ", ".join(functions) + "}")
(directory / "cases.h").write_text(
    "\n".join(header) + "\nstatic const Case cases[] = {\n" + ",\n".join(cases) + "\n};\n"
)
subprocess.run(
    ["/opt/homebrew/opt/llvm/bin/clang++", "-std=c++23", "-O3", "-mcpu=native", "-Wno-override-module",
     "-I", str(directory), "-I/opt/homebrew/opt/openblas/include", "-L/opt/homebrew/opt/openblas/lib",
     str(root / "lit/tensor/group_default_bench.cpp"), str(directory / "group0.ll"),
     str(directory / "group1.ll"), "-lopenblas", "-o", str(directory / "bench")],
    check=True,
)
if args.build_only:
    print(directory / "bench")
else:
    compilers = {"mim", "clang", "clang++", "cc1", "ninja", "cmake", "make", "gmake"}
    for attempt in range(61):
        activity = subprocess.check_output(["ps", "-Ao", "pid=,comm=,args="], text=True)
        busy = []
        for line in activity.splitlines():
            fields = line.split(maxsplit=2)
            if len(fields) >= 2:
                command = pathlib.Path(fields[1]).name
                if command in compilers or command.startswith("clang-"):
                    busy.append(line)
        if not busy:
            break
        if attempt == 60:
            raise RuntimeError("Other builds still active; benchmark not run:\n" + "\n".join(busy))
        time.sleep(5)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with (directory / "results.csv").open("w") as output:
        subprocess.run([str(directory / "bench")], env=environment, stdout=output, check=True)
    print((directory / "results.csv").read_text(), end="")
