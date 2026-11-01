import pathlib
import subprocess
import sys


mim, source, output, *reference = sys.argv[1:]
text = pathlib.Path(source).read_text().split("extern fun square")[0]
text = text.replace("plugin core;", "plugin core;\nplugin math;\nlet F32 = math.F32;")
text = text.replace("I64", "F32").replace("0F32", "0.0:F32")
text = text.replace("core.wrap.add 0 @0", "math.arith.add math.mode.fast")
text = text.replace("core.wrap.mul 0 @0", "math.arith.mul math.mode.fast")
for name, m, k, n in (("full", 8, 8, 8), ("tail", 13, 7, 11), ("skinny", 1, 7, 1), ("register_tail", 5, 9, 19)):
    for kind in ("rule", "hand"):
        config = "(4, 16, 16, 4, ff)"
        product = f"product_2d Ring @({m}, {k}, {n}) (buf a, buf b)"
        value = (
            f"mark @({m}, {k}, {n}) ({config}, {product})"
            if kind == "rule"
            else f"packed @({m}, {k}, {n}) ({config}, buf a, buf b, "
            "math.arith.add math.mode.fast, math.arith.mul math.mode.fast)"
        )
        text += (
            f"extern fun {kind}_{name} (a: «{m}, {k}; F32», b: «{k}, {n}; F32»): «{m}, {n}; F32» =\n"
            f"    return ({value});\n"
        )
text += """
extern fun selected_config (): «4; Nat» =
    let (mr, nr, nc, kc, _) = select_config (vec_width 4, 32, 262144, ff);
    return (mr, nr, nc, kc);
import opt;
extern lam _compile (): compile.Phase =
    compile.phases ff (compile.rule_app pack, opt._default_compile ());
"""
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
generated = directory / "probe.mim"
generated.write_text(text)
flags = ["-X", "tensor:reference-tiles=1"] if reference else []
subprocess.run([mim, *flags, "-p", "opt", str(generated), "-p", "ll"], cwd=directory, check=True)
