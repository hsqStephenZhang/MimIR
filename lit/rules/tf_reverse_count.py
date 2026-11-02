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
    match = re.search(r" = mul(?: nsw| nuw)* (i64|<(\d+) x i64>) %[^,]+, %", line)
    if match:
        lanes = int(match.group(2)) if match.group(2) else 1
        lines.append(f"  call void @record_multiply(i64 {lanes})")
lines.append("declare void @record_multiply(i64)")
destination.write_text("\n".join(lines) + "\n")
