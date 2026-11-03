import pathlib
import subprocess
import sys


mim, source, output, *flags = sys.argv[1:]
mim = str(pathlib.Path(mim).resolve())
text = pathlib.Path(source).read_text()
shapes = [(5, 7), (1, 11), (7, 1), (2, 7), (64, 7), (65, 7)]
benchmark = "benchmark" in flags
automatic = "automatic" in flags
if automatic and "all-shapes" not in flags:
    shapes = [(5, 7), (2, 7), (64, 7), (65, 7)]
row_override = next((int(flag.split("=", 1)[1]) for flag in flags if flag.startswith("rows=")), None)
if row_override is not None and row_override <= 0:
    raise ValueError("row tile must be positive")
if benchmark:
    shapes = [(1024, 1024), (4096, 1024), (1024, 4096)]
declarations = []
cases = []
for m, n in shapes:
    for kind, marker in (("soft", "mark_soft"), ("layer", "mark_layer")):
        rows = row_override if row_override is not None else 1 if benchmark else 2
        expression = f"row_{kind} (buf x)" if automatic else f"{marker} @({m}, {n}) ({rows}, buf x)"
        if automatic and "api" in flags and kind == "soft":
            expression = "softmax mode tt (buf x)"
        text += (
            f"extern fun {kind}_{m}_{n} (x: tensor.Tensor 2 ({m}, {n}) F32): tensor.Tensor 2 ({m}, {n}) F32 =\n"
            f"    return ({expression});\n"
        )
        if benchmark:
            first = "row_max (buf x)" if kind == "soft" else "row_average (ff, buf x)"
            reduce = "row_sum (buf x)" if kind == "soft" else "row_average (tt, buf x)"
            transform = "shifted_exp" if kind == "soft" else "subtract"
            finish = "divide" if kind == "soft" else "normalize"
            text += (
                f"extern fun {kind}_first_{m}_{n} (x: tensor.Tensor 2 ({m}, {n}) F32): tensor.Tensor 1 {m} F32 = return ({first});\n"
                f"extern fun {kind}_reduce_{m}_{n} (x: tensor.Tensor 2 ({m}, {n}) F32): tensor.Tensor 1 {m} F32 = return ({reduce});\n"
            )
            for name, operation in (("transform", transform), ("finish", finish)):
                text += (
                    f"extern fun {kind}_{name}_{m}_{n} (x: tensor.Tensor 2 ({m}, {n}) F32, y: tensor.Tensor 1 {m} F32): tensor.Tensor 2 ({m}, {n}) F32 =\n"
                    f"    return (with_rows ({operation}, buf x, buf y));\n"
                )
            prefix = f"{kind}_"
            suffix = f"_{m}_{n}"
            declarations.append(f'extern "C" float* {prefix}{m}_{n}(float*);')
            for name in ("first", "reduce", "transform", "finish"):
                arguments = "float*" if name in ("first", "reduce") else "float*, float*"
                declarations.append(f'extern "C" float* {prefix}{name}{suffix}({arguments});')
            functions = [f"{kind}_{m}_{n}"] + [
                f"{prefix}{name}{suffix}" for name in ("first", "transform", "reduce", "finish")
            ]
            cases.append(
                f'{{{m}, {n}, {rows}, {"true" if kind == "layer" else "false"}, '
                + ", ".join(functions) + "}"
            )
if not automatic:
    text += """
import opt;
extern lam _compile (): compile.Phase =
    compile.phases ff (compile.rule_app (group_soft, group_layer), opt._default_compile ());
"""
directory = pathlib.Path(output).resolve()
directory.mkdir(exist_ok=True)
generated = directory / "probe.mim"
generated.write_text(text)
if benchmark:
    (directory / "cases.h").write_text(
        "\n".join(declarations) + "\nCase cases[] = {\n" + ",\n".join(cases) + "\n};\n"
    )
options = ["-X", "tensor:reference-tiles=1"] if "reference" in flags else []
if automatic:
    options += ["-X", "tensor:strict=1"]
subprocess.run([mim, *options, "-p", "opt", str(generated), "-p", "ll"], cwd=directory, check=True)
