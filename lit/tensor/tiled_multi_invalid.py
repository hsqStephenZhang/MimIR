import pathlib
import subprocess
import sys

mim, output = sys.argv[1:]
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
for name, grid, tiles, shape, axes, message in [
    ("duplicate", "(5, 5)", "(2, 2)", "(5, 5)", "(0_2, 0_2)", "distinct result axis"),
    ("missing", "(5, 5)", "(2, 2)", "5", "(0_1, 0_1)", "distinct result axis"),
    ("extent", "5", "2", "(6, 3)", "0_2", "mapped grid extent"),
    ("zero_tile", "5", "0", "(5, 3)", "0_2", "positive tile sizes"),
]:
    rank = 1 if name == "missing" else 2
    g = 2 if name in ("duplicate", "missing") else 1
    source = directory / f"{name}.mim"
    source.write_text(
        "plugin tensor;\nuse tensor;\n"
        f"extern fun invalid (): tensor.Tensor {rank} {shape} I64 =\n"
        f"    return (tiled_multi @(1, {g}, I64, {rank}) ({shape}, {grid}, {tiles}) {axes} ()\n"
        f"        (λ (off ext: «{g}; Nat») (): tensor.Tensor {rank} (tile_shape @({rank}, {g}) ({shape}, {axes}, ext)) I64 =\n"
        f"            splat (tile_shape @({rank}, {g}) ({shape}, {axes}, ext), 0I64)));\n"
    )
    result = subprocess.run([mim, str(source)], capture_output=True, text=True)
    assert result.returncode == 1, (name, result.stderr)
    assert message in result.stderr, (name, result.stderr)
