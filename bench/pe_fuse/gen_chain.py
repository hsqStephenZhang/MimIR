"""Emit a Mim file with a depth-k GEMM chain X_l = X_{l-1}·W_l, once recursively fused and once layer by layer.

python gen_chain.py K M D TI TK [fused|layered|both] > chain.mim
"""
import sys

K, M, D, TI, TK = map(int, sys.argv[1:6])
WHICH = sys.argv[6] if len(sys.argv) > 6 else "both"

HDR = f"""plugin tensor;
plugin core;
plugin math;
plugin mem;
plugin buffer;
plugin btensor;
plugin affine;

let F32 = math.F32;
let Ix  = affine.Index;
let Bx  = buffer.Buf (2, ({M}, {D}), F32);
let Bw  = buffer.Buf (2, ({D}, {D}), F32);
let ij  = λ (o: «3; Ix»): «2; Ix» = (o#0₃, o#1₃);
let ik  = tensor.aff_proj @(3, 2) (0₃, 2₃);
let kj  = tensor.aff_proj @(3, 2) (2₃, 1₃);
let P   = tensor.Prod (2, F32);

fun mac (m: mem.M 0, acc: F32, (x y: F32))@tt: [mem.M 0, F32] = return (m, math.arith.add 0 (acc, math.arith.mul 0 (x, y)));
"""

ws = ", ".join(f"w{l}: Bw" for l in range(1, K + 1))


def tree(stage):
    leaf = "tensor.sched_leaf @(2, 3) 1"
    body = f"tensor.sched_stage @(2, 3) (0₂, {leaf})" if stage else leaf
    return f"tensor.sched_par @(2, 3) (({TI}, 0, {TK}), {body})"


def fused():
    out = [f"extern fun fused (m: mem.M 0, a: Bx, {ws}): [mem.M 0, Bx] ="]
    for l in range(1, K + 1):
        sig = f"λ (lo ext: «2; Nat») (m: mem.M 0): [mem.M 0, buffer.Buf (2, ext, F32)] ="
        maps = ("(tensor.aff_offset @(3, 2) ((0₃, 2₃), (lo#0₂, 0)),\n"
                "              tensor.aff_offset @(3, 2) ((2₃, 1₃), (0, lo#1₂)))")
        if l == 1:
            out += [f"    let fill{l} = {sig}",
                    f"        tensor.run_levels_with @(2, F32, 2, 3, (F32, F32), (2, 2), (({M}, {D}), ({D}, {D})))",
                    f"            (ext, (ext#0₂, ext#1₂, {D})) (mac, 0.0:F32) ij",
                    f"            ({maps},",
                    f"             (P#Plain, P#Plain))",
                    f"            ({tree(False)}, (ext#0₂, ext#1₂, {D}), (ff, ff, ff))",
                    f"            (m, (a, w{l}));"]
        else:
            out += [f"    let fill{l} = {sig}",
                    f"        let (m, dummy) = buffer.alloc (2, (2, 2), F32) m;",
                    f"        let (m, x) = tensor.run_levels_with @(2, F32, 2, 3, (F32, F32), (2, 2), ((2, 2), ({D}, {D})))",
                    f"            (ext, (ext#0₂, ext#1₂, {D})) (mac, 0.0:F32) ij",
                    f"            ({maps},",
                    f"             (P#Fill fill{l - 1}, P#Plain))",
                    f"            ({tree(True)}, (ext#0₂, ext#1₂, {D}), (ff, ff, ff))",
                    f"            (m, (dummy, w{l}));",
                    f"        (buffer.free (2, (2, 2), F32) (m, dummy), x);"]
    out += [f"    return (fill{K} ((0, 0), ({M}, {D})) m);"]
    return "\n".join(out)


def layered():
    out = [f"extern fun layered (m: mem.M 0, a: Bx, {ws}): [mem.M 0, Bx] ="]
    prev = "a"
    for l in range(1, K + 1):
        out += [f"    let (m, x{l}) = tensor.run_levels @(2, F32, 2, 3, (F32, F32), (2, 2), (({M}, {D}), ({D}, {D})))",
                f"        (({M}, {D}), ({M}, {D}, {D})) (mac, 0.0:F32) ij (ik, kj)",
                f"        ({tree(False)}, ({M}, {D}, {D}), (ff, ff, ff)) (m, ({prev}, w{l}));"]
        if l > 1:
            out += [f"    let m = buffer.free (2, ({M}, {D}), F32) (m, {prev});"]
        prev = f"x{l}"
    out += [f"    return (m, {prev});"]
    return "\n".join(out)


print(HDR)
if WHICH in ("fused", "both"):
    print(fused())
if WHICH in ("layered", "both"):
    print(layered())
