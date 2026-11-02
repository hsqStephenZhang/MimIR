import pathlib
import subprocess
import sys


mim, source, output, *flags = sys.argv[1:]
mim = str(pathlib.Path(mim).resolve())
text = pathlib.Path(source).read_text()
shapes = [(5, 7), (1, 11), (7, 1), (0, 7)]
benchmark = "benchmark" in flags
if benchmark:
    shapes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (1000, 1000), (513, 129)]
for m, n in shapes:
    for kind, marker in (("soft", "mark_soft"), ("layer", "mark_layer")):
        rows = max(1, min(16, 32768 // (n * 4))) if benchmark else 2
        text += (
            f"extern fun {kind}_{m}_{n} (x: «{m}, {n}; F32»): «{m}, {n}; F32» =\n"
            f"    return ({marker} @({m}, {n}) ({rows}, buf x));\n"
        )
        if benchmark:
            text += (
                f"extern fun {kind}_unfused_{m}_{n} (x: «{m}, {n}; F32»): «{m}, {n}; F32» =\n"
                f"    return (row_{kind} @({m}, {n}) (buf x));\n"
            )
text += """
import opt;
extern lam _compile (): compile.Phase =
    compile.phases ff (compile.rule_app (group_soft, group_layer), opt._default_compile ());
"""
directory = pathlib.Path(output).resolve()
directory.mkdir(exist_ok=True)
generated = directory / "probe.mim"
generated.write_text(text)
options = ["-X", "tensor:reference-tiles=1"] if "reference" in flags else []
subprocess.run([mim, *options, "-p", "opt", str(generated), "-p", "ll"], cwd=directory, check=True)
