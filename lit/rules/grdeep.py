import pathlib
import re
import subprocess
import sys


mim, temporary = sys.argv[1:]
directory = pathlib.Path(f"{temporary}.dir")
directory.mkdir()
source = directory / "deep.mim"
try:
    for count in (400,):
        source.write_text(
            "plugin compile;\n"
            "axm q: Nat -> Nat;\n"
            "axm v: Nat -> Nat;\n"
            "axm s: Nat -> Nat;\n"
            "rule replace (x: Nat): q x => v x;\n"
            "extern lam result (x: Nat): Nat = q ("
            + "s (" * count + "x" + ")" * count + ");\n"
            "extern lam _compile (): compile.Phase = "
            "compile.graph_rule_app (replace, (q, v, s));\n"
        )
        result = subprocess.run(
            [mim, str(source), "-o", "-"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        body = result.stdout.split("extern lam result", 1)[1]
        assert re.search(r"\bv ", body), body[:500]
        assert not re.search(r"\bq ", body), body[:500]
        assert len(re.findall(r"\bs(?:\s|$)", body)) == count, body[:2000]
        print(f"{count}: rewritten, all chain operations retained")
finally:
    source.unlink()
    directory.rmdir()
