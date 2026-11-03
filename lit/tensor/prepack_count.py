import pathlib
import sys

source, destination = map(pathlib.Path, sys.argv[1:3])
text = source.read_text()
if "@aligned_alloc(" not in text:
    raise RuntimeError("embedded persistent allocation call was not emitted")
destination.write_text(text.replace("@aligned_alloc(", "@record_pack("))
