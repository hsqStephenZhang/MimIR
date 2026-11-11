import pathlib
import subprocess
import sys


mim, source, output, *reference = sys.argv[1:]
post = "post" in reference
workloads = "workloads" in reference
text = pathlib.Path(source).read_text().split("extern fun square")[0]
text = text.replace("plugin core;", "plugin core;\nplugin math;\nlet F32 = math.F32;")
text = text.replace("I64", "F32").replace("0F32", "0.0:F32")
text = text.replace("core.wrap.add 0 @0", "math.arith.add math.mode.fast")
text = text.replace("core.wrap.mul 0 @0", "math.arith.mul math.mode.fast")
text = text.replace("core.wrap.add 0 (", "math.arith.add math.mode.fast (")
text = text.replace("core.icmp.sl", "math.cmp.l math.mode.fast")
shapes = [("full", 8, 8, 8), ("tail", 13, 7, 11), ("skinny", 1, 7, 1), ("register_tail", 5, 9, 19)]
if post:
    shapes.append(("nc_tail", 5, 11, 21))
if workloads:
    shapes = []
    text += pathlib.Path(source).with_name("blis_workloads.mim").read_text()
for name, m, k, n in shapes:
    for kind in ("rule", "hand"):
        config = "(4, 16, 16, 4, ff)"
        product = f"product_2d Ring @({m}, {k}, {n}) (buf a, buf b)"
        value = (
            f"mark @({m}, {k}, {n}) ({config}, {product})"
            if kind == "rule"
            else f"packed @({m}, {k}, {n}) ({config}, buf a, buf b, "
            "math.arith.add math.mode.fast, math.arith.mul math.mode.fast)"
        )
        if post:
            value = (
                f"mark_post @({m}, {k}, {n}) ({config}, {product}, buf bias)"
                if kind == "rule"
                else f"packed_with_post @({m}, {k}, {n}) ({config}, buf a, buf b, "
                "math.arith.add math.mode.fast, math.arith.mul math.mode.fast, buf bias, bias_relu, tt)"
            )
        bias = f", bias: tensor.Tensor 1 {n} F32" if post else ""
        text += (
            f"extern fun {kind}_{name} (a: tensor.Tensor 2 ({m}, {k}) F32, b: tensor.Tensor 2 ({k}, {n}) F32{bias}): tensor.Tensor 2 ({m}, {n}) F32 =\n"
            f"    return ({value});\n"
        )
text += """
extern fun selected_config (): «4; Nat» =
    let (mr, nr, nc, kc, _) = select_config (vec_width 4, 32, 262144, ff);
    return (mr, nr, nc, kc);
import opt;
extern lam _compile (): compile.Phase =
    compile.phases ff (compile.rule_app (pack, pack_post), opt._default_compile ());
"""
if workloads:
    text = text.replace(
        "compile.rule_app (pack, pack_post)",
        "compile.phases tt (compile.rule_app (pack_post, pack, tensor.group_outer (tensor.Tensor 2 (5, 11) F32, 2), "
        "tensor.fuse_producer (F32, 1), tensor.fuse_producer (F32, 2)), "
        "tensor.fuse_region_inputs (0, 1, ff), compile.cleanup)",
    )
if "schedule" in reference:
    text = text.replace("opt._default_compile ()", "compile.phases ff (tensor.tile_fuse, compile.cleanup)")
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
generated = directory / "probe.mim"
generated.write_text(text)
flags = ["-X", "tensor:reference-tiles=1"] if "reference" in reference else []
backend = ["-o", "-"] if "schedule" in reference else ["-p", "ll"]
subprocess.run([mim, *flags, "-p", "opt", str(generated), *backend], cwd=directory, check=True)
