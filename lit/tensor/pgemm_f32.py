import pathlib
import subprocess
import sys


mim, source, output = sys.argv[1:]
text = pathlib.Path(source).read_text().split("extern fun square")[0]
text = text.replace("plugin core;", "plugin core;\nplugin math;\nlet F32 = math.F32;")
text = text.replace("I64", "F32").replace("0F32", "0.0:F32")
text = text.replace("core.wrap.add 0 @0", "math.arith.add math.mode.fast")
text = text.replace("core.wrap.mul 0 @0", "math.arith.mul math.mode.fast")
text += """
extern fun vector_leaf (a: «8, 64; F32», b: «64, 32; F32»): «8, 32; F32» =
    return (mark @(8, 64, 32, (4, 16, 32))
        (leaf_schedule, product_2d Ring @(8, 64, 32) (buf a, buf b)));
import opt;
extern lam _compile (): compile.Phase =
    compile.phases ff (compile.rule_app pack, opt._default_compile ());
"""
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
generated = directory / "probe.mim"
generated.write_text(text)
subprocess.run([mim, "-p", "opt", str(generated), "-p", "ll"], cwd=directory, check=True)
