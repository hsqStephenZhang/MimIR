import pathlib
import subprocess
import sys


mim, filename, temporary, *options = sys.argv[1:]
text = pathlib.Path(filename).read_text()
if "normalization" in options:
    text += """
extern fun grouped_softmax (x: «5, 7; F32»): «5, 7; F32» = return (row_soft (buf x));
extern fun grouped_layernorm (x: «5, 7; F32»): «5, 7; F32» = return (row_layer (buf x));
"""
directory = pathlib.Path(f"{temporary}.dir")
directory.mkdir()
generated = directory / "probe.mim"
phases = "tensor.lower_tensor, tensor.fuse_tensor, tensor.group"
if "twice" in options:
    phases += ", tensor.group"
if "report" in options:
    phases += ", tensor.report_groups"
if "realize" in options:
    phases += ", tensor.group_rules"
if "gpu" in options:
    text += "\nplugin gpu;\n"
    phases += ", tensor.group_rules"
if "buffer" in options:
    phases += ", tensor.lower_to_mem"
if "lower" in options:
    text += "\nplugin opt;\n"
    phases += ", opt._default_compile ()"
try:
    generated.write_text(
        text + "\nextern lam _compile (): compile.Phase =\n"
        f"    compile.phases ff ({phases}, compile.cleanup);\n"
    )
    arguments = [mim, str(generated), "-o", "-", "-V", "-V", "-V", "-V"]
    arguments += [arg for arg in options if arg.startswith("-")]
    if "lower" in options:
        arguments += ["-p", "ll", "-X", f"ll:o={temporary}.ll"]
    result = subprocess.run(arguments, text=True, capture_output=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)
finally:
    generated.unlink()
    directory.rmdir()
