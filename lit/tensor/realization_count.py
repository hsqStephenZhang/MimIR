import pathlib
import re
import subprocess
import sys


source, destination = map(pathlib.Path, sys.argv[1:3])
exponentials = "--exp" in sys.argv[3:]
centered = "--center" in sys.argv[3:]
callback = "record_exp" if exponentials else "record_center" if centered else "record_multiply"
optimization = "-O0" if "--unoptimized" in sys.argv[3:] else "-O3"
optimized = subprocess.run(
    ["clang", optimization, "-S", "-emit-llvm", "-Wno-override-module", str(source), "-o", "-"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
).stdout
lines = []
for line in optimized.splitlines():
    lines.append(line)
    operation = (
        " call " in line and "@llvm.exp." in line
        if exponentials
        else " fsub " in line
        if centered
        else " fmul " in line
    )
    if operation:
        vector = re.search(r"<(\d+) x float>", line)
        lanes = int(vector.group(1)) if vector else 1
        lines.append(f"  call void @{callback}(i64 {lanes})")
lines.append(f"declare void @{callback}(i64)")
destination.write_text("\n".join(lines) + "\n")
