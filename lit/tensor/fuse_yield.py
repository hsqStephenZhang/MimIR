import pathlib
import subprocess
import sys


mim, source, temporary, mode = sys.argv[1:]
generated = pathlib.Path(temporary + ".probe.mim")
budget = 1 if mode == "budget" else 3
pipeline = f"compile.phases tt (tensor.fuse_and_yield {budget}, compile.cleanup)"
if mode == "plain":
    pipeline = "compile.cleanup"
if mode != "structure":
    pipeline = f"compile.phases ff ({pipeline}, opt._default_compile ())"
generated.write_text(
    pathlib.Path(source).read_text()
    + "\nplugin opt;\nextern lam _compile (): compile.Phase = "
    + pipeline
    + ";\n"
)
arguments = [mim, str(generated), "-X", "tensor:group=0", "-VVVV"]
if mode == "structure":
    arguments += ["-o", "-"]
else:
    arguments += ["-p", "ll", "-X", f"ll:o={temporary}.ll"]
try:
    subprocess.run(arguments, check=True)
finally:
    generated.unlink()
