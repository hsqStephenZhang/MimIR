import pathlib
import subprocess
import sys

mim, output = sys.argv[1:]
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
cases = [
    (
        "tiled",
        "tiled (8, 4) () (λ (off ext: Nat) (): «ext; I64» = "
        "slice @(I64, 1) 8 (off, 1, ext) outer_tensor)",
        "tensor.slice",
    ),
    (
        "bypass_parameter",
        "tiled (8, 4) outer_tensor (λ (off ext: Nat) (p): «ext; I64» = "
        "slice @(I64, 1) 8 (off, 1, ext) outer_tensor)",
        "tensor.slice",
    ),
    (
        "tile_fold",
        "tile_fold (2, 1) local_tensor () "
        "(λ (acc: «8; I64», off: Nat) (): «8; I64» = "
        "slice @(I64, 1) 8 (0, 1, 8) outer_tensor)",
        "tensor.tile_fold",
    ),
    (
        "tiled_n",
        "tiled_n @(1, 1, I64, 1) (8, 2) "
        "(λ (point: Nat): TileBox 1 = (core.nat.mul (point, 4), 4)) () "
        "(λ (point: Nat) (): «4; I64» = "
        "slice @(I64, 1) 8 (core.nat.mul (point, 4), 1, 4) outer_tensor)",
        "tensor.slice",
    ),
    (
        "mixed_inputs",
        "tiled (8, 4) local_tensor (λ (off ext: Nat) (local_tensor): «ext; I64» = "
        "slice @(I64, 1) 8 (off, 1, ext) "
        "(map_reduce @2 (8, 8) (add, 0I64) (affine.id @1) "
        "(affine.id @1, affine.id @1) (local_tensor, outer_tensor)))",
        "tensor.map_reduce_post",
    ),
    (
        "nested",
        "tiled (8, 4) outer_tensor (λ (off ext: Nat) (outer_tensor): «ext; I64» = "
        "tiled (ext, 2) () (λ (inner_off inner_ext: Nat) (): «inner_ext; I64» = "
        "slice @(I64, 1) 8 (inner_off, 1, inner_ext) outer_tensor))",
        "tensor.slice",
    ),
]
for name, expression, operation in cases:
    source = directory / f"{name}.mim"
    source.write_text(
        "plugin tensor;\nplugin core;\nplugin affine;\nuse tensor;\n"
        "fun add (_: I64, (a b: I64))@tt: I64 = return (core.wrap.add 0 (a, b));\n"
        "extern fun invalid (outer_tensor local_tensor: «8; I64»): «8; I64» =\n"
        f"    return ({expression});\n"
    )
    result = subprocess.run([mim, str(source)], capture_output=True, text=True)
    assert result.returncode == 1, (name, result.stderr)
    assert f"tensor region operation {operation} reads outer tensor input" in result.stderr, (name, result.stderr)
    assert "outer_tensor" in result.stderr, (name, result.stderr)
    assert "pass it explicitly" in result.stderr, (name, result.stderr)
