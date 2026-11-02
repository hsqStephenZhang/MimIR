#!/usr/bin/env python3
"""Compare identical row programs through automatic, explicit, and tensor-API lowering."""

import argparse
import os
import pathlib
import re
import subprocess
import time


root = pathlib.Path(__file__).resolve().parent.parent
clang = "/opt/homebrew/opt/llvm/bin/clang++"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=pathlib.Path, default=root / "build/task21-row-bench")
parser.add_argument("--build-only", action="store_true")
parser.add_argument("--run-only", action="store_true")
args = parser.parse_args()
directory = args.directory.resolve()
directory.mkdir(parents=True, exist_ok=True)


def wait_idle():
    for attempt in range(61):
        busy = []
        for line in subprocess.check_output(["ps", "-Ao", "pid=,comm="], text=True).splitlines():
            command = pathlib.Path(line.split(maxsplit=1)[1]).name
            if command in {"mim", "clang", "clang++", "ninja", "cmake", "make", "gmake"} or command.startswith("clang-"):
                busy.append(line)
        if not busy:
            return
        if attempt == 60:
            raise RuntimeError("Other builds still active:\n" + "\n".join(busy))
        time.sleep(5)


def benchmark():
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for executable, result in (("count", "counts.csv"), ("bench", "results.csv")):
        wait_idle()
        with (directory / result).open("w") as output:
            subprocess.run([str(directory / executable)], env=environment, stdout=output, check=True)
    wait_idle()
    print((directory / "results.csv").read_text(), end="")


if args.run_only:
    benchmark()
    raise SystemExit(0)
shapes = [(1024, 1024), (4096, 1024), (1024, 4096)]
canonical = (root / "lit/tensor/row_norm.mim").read_text()
legacy = (root / "lit/tensor/group_default.mim").read_text().split("extern fun gemm")[0]
api = """
lam api_layer {M N: Nat} (x: «M, N; F32»): «M, N; F32» =
    let divisor = math.conv.u2f math.f32 (core.idx 0 0 N);
    lam average (x: «M, N; F32»): «M; F32» =
        unary (λ (v: F32): F32 = math.arith.div mode (v, divisor))
            (reduce (math.arith.add mode, 0.0:F32) tt x);
    let mean = average x;
    let centered = binary_ax (math.arith.sub mode) tt (x, mean);
    let squared = unary (λ (v: F32): F32 = math.arith.mul mode (v, v)) centered;
    let variance = average squared;
    binary_ax (λ (v variance: F32): F32 =
        math.arith.div mode (v, math.rt.sq mode (math.arith.add mode (variance, 0.00001:F32))))
        tt (centered, variance);
"""
modes = ("unfused", "automatic", "explicit", "api", "legacy_default", "legacy_stages")
declarations, cases, modules = [], [], []
for index, mode in enumerate(modes):
    text = legacy if mode == "legacy_default" else canonical + api
    for m, n in shapes:
        for kind in ("soft", "layer"):
            name = f"v{index}_{kind}_{m}_{n}"
            expression = f"row_{kind} (buf x)"
            if mode == "explicit":
                expression = f"mark_{kind} @({m}, {n}) (1, buf x)"
            elif mode == "api":
                expression = "softmax mode tt (buf x)" if kind == "soft" else "api_layer (buf x)"
            if mode == "legacy_stages":
                first = "row_max (buf x)" if kind == "soft" else "row_average (ff, buf x)"
                reduce = "row_sum (buf x)" if kind == "soft" else "row_average (tt, buf x)"
                transform = "shifted_exp" if kind == "soft" else "subtract"
                finish = "divide" if kind == "soft" else "normalize"
                for suffix, expr, signature, result in (
                    ("first", first, "", f"«{m}; F32»"),
                    ("reduce", reduce, "", f"«{m}; F32»"),
                    ("transform", f"with_rows ({transform}, buf x, buf y)", f", y: «{m}; F32»", f"«{m}, {n}; F32»"),
                    ("finish", f"with_rows ({finish}, buf x, buf y)", f", y: «{m}; F32»", f"«{m}, {n}; F32»"),
                ):
                    text += f"extern fun {name}_{suffix} (x: «{m}, {n}; F32»{signature}): {result} = return ({expr});\n"
                    signature = "float*" if not signature else "float*, float*"
                    declarations.append(f'extern "C" float* {name}_{suffix}({signature});')
                functions = ", ".join(f"{name}_{suffix}" for suffix in ("first", "transform", "reduce", "finish"))
                cases.append(f'{{"{mode}", "{kind}", {m}, {n}, nullptr, {functions}}}')
            else:
                text += f"extern fun {name} (x: «{m}, {n}; F32»): «{m}, {n}; F32» = return ({expression});\n"
                declarations.append(f'extern "C" float* {name}(float*);')
                cases.append(f'{{"{mode}", "{kind}", {m}, {n}, {name}}}')
    if mode == "explicit":
        text += """
import opt;
extern lam _compile (): compile.Phase =
    compile.phases ff (compile.rule_app (group_soft, group_layer), opt._default_compile ());
"""
    source = directory / f"v{index}.mim"
    source.write_text(text)
    module = directory / f"{mode}.ll"
    group = [] if mode in ("automatic", "api") else ["-X", "tensor:group=0"]
    with (directory / f"{mode}.log").open("w") as log:
        subprocess.run(
            [str(root / "build/bin/mim"), "-p", "opt", "-p", "ll", *group,
             "-X", f"ll:o={module}", "--profile", "summary", "--output-profile",
             str(directory / f"{mode}.profile"), str(source)], stdout=log, stderr=log, check=True,
        )
    modules.append(str(module))
(directory / "cases.h").write_text(
    "\n".join(declarations) + "\nstatic const Case cases[] = {\n" + ",\n".join(cases) + "\n};\n"
)
common = [clang, "-std=c++23", "-O3", "-mcpu=native", "-Wno-override-module", "-I", str(directory),
          str(root / "scripts/bench-row-pipeline.cpp")]
subprocess.run([*common, *modules, "-o", str(directory / "bench")], check=True)
instrumented = []
for module in modules:
    optimized = subprocess.check_output(
        [clang, "-O3", "-mcpu=native", "-S", "-emit-llvm", "-Wno-override-module", module, "-o", "-"], text=True,
    )
    lines = []
    for line in optimized.replace("@malloc(", "@record_malloc(").replace("@free(", "@record_free(").splitlines():
        lines.append(line)
        if " alloca " in line:
            lines.append("  call void @record_stack()")
        if " call " in line and "@llvm.exp." in line:
            vector = re.search(r"<(\d+) x float>", line)
            lines.append(f"  call void @record_exp(i64 {int(vector.group(1)) if vector else 1})")
    lines.extend(["declare void @record_stack()", "declare void @record_exp(i64)"])
    path = pathlib.Path(module).with_suffix(".count.ll")
    path.write_text("\n".join(lines) + "\n")
    instrumented.append(str(path))
subprocess.run([*common, "-O0", "-DCOUNTS", *instrumented, "-o", str(directory / "count")], check=True)
if args.build_only:
    print(directory / "bench")
else:
    benchmark()
