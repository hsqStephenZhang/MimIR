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
    pipeline = f"compile.phases {fixed_point} ({phase}, compile.cleanup)"
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
