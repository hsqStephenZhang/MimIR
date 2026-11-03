import pathlib
import re
import sys

source, destination = map(pathlib.Path, sys.argv[1:3])
text = source.read_text()
if "@aligned_alloc(" not in text:
    raise RuntimeError("embedded persistent allocation call was not emitted")
text = text.replace("@aligned_alloc(", "@record_pack(").replace("@free(", "@record_free(")
lines = []
stores = 0
for line in text.splitlines():
    lines.append(line)
    store = re.match(r"\s*store float [^,]+, (?:float(?: addrspace\(0\))?\*|ptr) (%[-\w.$]+)(?:,.*)?$", line)
    if store:
        stores += 1
        lines.append(f"  call void @record_packed_write(ptr {store[1]})")
if not stores:
    raise RuntimeError("no packed fill stores were instrumented")
lines.append("declare void @record_packed_write(ptr)")
destination.write_text("\n".join(lines) + "\n")
