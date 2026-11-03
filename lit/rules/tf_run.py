import pathlib
import re
import subprocess
import sys


mim, filename, temporary, frontend, *strategy = sys.argv[1:]
fixed_point = strategy[0] if strategy else "ff"
assert fixed_point in ("ff", "tt")
lower = len(strategy) > 1 and strategy[1] == "lower"
source = pathlib.Path(filename)
text = source.read_text()
rules = re.search(r"^// TF-RULES: (.+)$", text, re.MULTILINE).group(1)
pure = re.search(r"^// TF-PURE: (.+)$", text, re.MULTILINE).group(1)
pure = ", ".join(
    f'"{op.strip()}"' if "." in op else op.strip() for op in pure.split(",")
)
phase = (
    f"compile.rule_app {rules}"
    if frontend == "rule"
    else f"compile.graph_rule_app ({rules}, ({pure}))"
)
schemas = re.search(r"^// TF-SCHEMAS: (.+)$", text, re.MULTILINE)
if schemas:
    phase = (
        f"compile.rule_app_with ({rules}, {schemas.group(1)})"
        if frontend == "rule"
        else f"compile.graph_rule_app_with ({rules}, ({pure}), {schemas.group(1)})"
    )
directory = pathlib.Path(f"{temporary}.{frontend}.dir")
directory.mkdir()
generated = directory / "probe.mim"
try:
    region_fusion = re.search(r"^// TF-REGION-FUSION: (.+)$", text, re.MULTILINE)
    if region_fusion:
        backend = "tt" if frontend == "graph" else "ff"
        phase = f"{phase}, tensor.fuse_region_inputs ({region_fusion.group(1)}, {backend})"
    pipeline = f"compile.phases {fixed_point} ({phase}, compile.cleanup)"
    if len(strategy) > 1 and strategy[1] == "tensor":
        pipeline = f"compile.phases ff ({pipeline}, tensor.lower_tensor, compile.cleanup)"
    if lower:
        pipeline = f"compile.phases ff ({pipeline}, opt._default_compile ())"
    generated.write_text(
        text
        + ("\nplugin opt;\n" if lower else "")
        + "\nextern lam _compile (): compile.Phase =\n"
        + f"    {pipeline};\n"
    )
    arguments = [mim, str(generated), "-V", "-V", "-V", "-V", "-o", "-"]
    if lower:
        arguments += ["-p", "ll", "-X", f"ll:o={temporary}.{frontend}.ll"]
        if len(strategy) > 2 and strategy[2] == "reference":
            arguments += ["-X", "tensor:reference-tiles=1"]
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)
finally:
    generated.unlink()
    directory.rmdir()
