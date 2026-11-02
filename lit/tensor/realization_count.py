import pathlib
import re
import subprocess
import sys


source, destination = map(pathlib.Path, sys.argv[1:])
optimized = subprocess.run(
    ["clang", "-O3", "-S", "-emit-llvm", "-Wno-override-module", str(source), "-o", "-"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
).stdout
lines = []
for line in optimized.splitlines():
    lines.append(line)
    if " fmul " in line:
        vector = re.search(r"<(\d+) x float>", line)
        lanes = int(vector.group(1)) if vector else 1
        lines.append(f"  call void @record_multiply(i64 {lanes})")
lines.append("declare void @record_multiply(i64)")
destination.write_text("\n".join(lines) + "\n")
