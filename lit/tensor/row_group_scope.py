import pathlib
import subprocess
import sys


mim, source, output, mode = sys.argv[1:]
text = pathlib.Path(source).read_text()
values = """
    let maximum = row_max (buf x);
    let exp = retained (with_rows (shifted_exp, buf x, maximum));
    let sum = row_sum exp;
    let result = with_rows (divide, exp, sum);
"""
if mode == "permuted":
    values = values.replace(
        "let sum = row_sum exp;",
        """let exp = stage @(F32, 2, 0) (5, 7) ((), (), (tt, ff), 0.0:F32) 0 exp;
    let exp = stage @(F32, 2, 0) (7, 5) ((), (), (tt, ff), 0.0:F32) 0 exp;
    let sum = row_sum exp;""",
    )
if mode == "cross":
    values = values.replace(
        "let sum = row_sum exp;",
        """let exp = map_reduce ((5, 7), (5, 7)) (copy, 0.0:F32) (affine.id @2)
        ((λ (i j: affine.Index): «2; affine.Index» = (affine.semiop.rem (affine.op.add (i, affine.lit 1), 5), j)),) exp;
    let sum = row_sum exp;""",
    )
    text += "\nfun copy (_ x: F32)@tt: F32 = return x;\n"
result_type = "[tensor.Tensor 2 (5, 7) F32, tensor.Tensor 2 (5, 7) F32]" if mode == "escape" else "tensor.Tensor 2 (5, 7) F32"
result = "(result, exp)" if mode == "escape" else "result"
text += f"\nextern fun probe (x: tensor.Tensor 2 (5, 7) F32): {result_type} =\n{values}    return {result};\n"
text += """
extern lam _compile (): compile.Phase =
    compile.phases ff (tensor.lower_tensor, tensor.fuse_tensor, tensor.group, compile.cleanup);
"""
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
generated = directory / "probe.mim"
try:
    generated.write_text(text)
    subprocess.run([mim, str(generated), "-o", "-", "-V", "-V", "-V", "-V"], check=True)
finally:
    generated.unlink()
    directory.rmdir()
