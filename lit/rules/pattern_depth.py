import re
import subprocess
import sys
from pathlib import Path

compiler, prefix = sys.argv[1:]
for count in (32, 256, 2048):
    directory = Path(f"{prefix}.{count}")
    directory.mkdir(exist_ok=True)
    source = directory / "pdepth.mim"
    output = directory / "output.mim"
    operands = ", ".join(["x"] + [f"x+{i}" for i in range(1, count)])
    source.write_text(
        f"""plugin compile;
plugin core;
use core.ops.n;
axm mark: «{count}; Nat» → Nat;
rule remove (x: Nat): mark ({operands}) => x;
extern lam result (x: Nat): Nat = mark ({operands});
extern lam _compile (): compile.Phase = compile.rule_app remove;
"""
    )
    subprocess.run([compiler, str(source), "-o", str(output)], check=True)
    assert re.search(r"extern lam result \(x: Nat\): Nat =\s+x;", output.read_text()), count
