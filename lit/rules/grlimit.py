import pathlib
import re
import subprocess
import sys


mim, temporary = sys.argv[1:]
count = 1001
values = range(10000, 10000 + count)
calls = ", ".join(f"q {i}" for i in values)
directory = pathlib.Path(f"{temporary}.dir")
for fixed_point, expected in (("ff", 1), ("tt", count)):
    directory.mkdir()
    source = directory / f"limit{fixed_point}.mim"
    source.write_text(
        "plugin compile;\n"
        "axm q: Nat -> Nat;\n"
        "axm v: Nat -> Nat;\n"
        "rule replace (x: Nat): q x => v x;\n"
        f"extern lam result (): «{count}; Nat» = ({calls});\n"
        "extern lam _compile (): compile.Phase =\n"
        f"    compile.phases {fixed_point} "
        "(compile.graph_rule_app (replace, (q, v)), compile.cleanup);\n"
    )
    try:
        result = subprocess.run(
            [mim, str(source), "-o", "-"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        output = result.stdout
        rewritten = re.findall(r"\bv (\d+)\b", output)
        retained = re.findall(r"\bq (\d+)\b", output)
        assert len(rewritten) == expected, (fixed_point, len(rewritten), expected)
        assert len(retained) == count - expected, (fixed_point, len(retained))
        assert set(map(int, rewritten + retained)) == set(values)
        print(f"{fixed_point}: {len(rewritten)} rewrites, {len(retained)} retained")
    finally:
        source.unlink()
        directory.rmdir()
