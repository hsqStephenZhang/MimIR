import pathlib
import re
import subprocess
import sys


mim, filename, temporary, frontend = sys.argv[1:]
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
directory = pathlib.Path(f"{temporary}.{frontend}.dir")
directory.mkdir()
generated = directory / "probe.mim"
try:
    generated.write_text(
        text
        + "\nextern lam _compile (): compile.Phase =\n"
        + f"    compile.phases ff ({phase}, compile.cleanup);\n"
    )
    result = subprocess.run(
        [mim, str(generated), "-V", "-V", "-V", "-V", "-o", "-"],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)
finally:
    generated.unlink()
    directory.rmdir()
