import pathlib
import subprocess
import sys

mim, output = sys.argv[1:]
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
cases = [
    ("(0, 3)", "3", "2", "overlap"),
    ("(point * 2, 1)", "4", "2", "do not cover"),
    ("(point * 2, 3)", "4", "2", "outside"),
    ("(0, 1)", "1", "0", "do not cover"),
]
for box, shape, grid, expected in cases:
    source = directory / "probe.mim"
    source.write_text(
        "plugin tensor;\nplugin core;\nuse tensor;\nuse core.ops.n;\n"
        f"extern fun invalid (): «{shape}; I64» =\n"
        f"    return (tiled_n @(1, 1, I64, 1) ({shape}, {grid})\n"
        f"        (λ (point: Nat): TileBox 1 = {box})\n"
        "        (λ (point: Nat): «"
        f"({box})#tt"
        "; I64» = splat ("
        f"({box})#tt, 0I64)));\n"
    )
    for flags in ([], ["-X", "tensor:reference-tiles=1"]):
        result = subprocess.run([mim, "-p", "opt", *flags, str(source)], capture_output=True, text=True)
        assert result.returncode == 1, result.stderr
        assert expected in result.stderr, result.stderr
