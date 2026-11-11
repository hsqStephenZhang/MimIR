import re
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text().splitlines()
ins = []
weights = {}
for loop in s:
    m = re.match(r"\s*([0-9.]+)\s*:\s*([0-9a-f]+):\s*(.*)", loop)
    if m:
        a = int(m[2], 16)
        weights[a] = float(m[1])
        ins.append((a, m[3]))
loops = []
for a, i in ins:
    m = re.match(r"j\w+\s+([0-9a-f]+)", i)
    if not m:
        continue
    t = int(m[1], 16)
    if t >= a:
        continue
    body = [(x, c) for x, c in ins if t <= x <= a]
    fma = [c for _, c in body if "vfmadd" in c]
    if not fma:
        continue
    loops.append(
        dict(
            start=t,
            end=a,
            scalar=sum("ss" in c.split()[0] for c in fma),
            vector=sum("ps" in c.split()[0] for c in fma),
            percent=sum(weights[x] for x, _ in body),
        )
    )
inner = [
    loop
    for loop in loops
    if not any(
        loop["start"] <= o["start"] and o["end"] <= loop["end"] and loop != o
        for o in loops
    )
]
print(json.dumps(inner, indent=2))
print(
    "scalar",
    sum(loop["percent"] for loop in inner if loop["scalar"] and not loop["vector"]),
    "vector",
    sum(loop["percent"] for loop in inner if loop["vector"] and not loop["scalar"]),
)
