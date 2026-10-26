import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


@dataclass
class Case:
    name: str
    family: str
    na: int
    nb: int
    no: int
    body: str
    width: int = 1
    height: int = 1
    depth: int = 1
    strategy: str = "one output per thread"
    groups: int = 0
    shared_n: int = 256
    shared_type: str = "F32"
    a_type: str = "F32"
    b_type: str = "F32"
    out_type: str = "F32"


def elementwise(name, expression, n=4099):
    return Case(
        name, name, n, n, n,
        f"""let (g, x) = load {n} (g, a, i);
    let (g, y) = load {n} (g, b, i);
    return (g, {expression});""",
    )


def cases():
    result = [
        elementwise("add", "add (x, y)"),
        elementwise("mul", "mul (x, y)"),
        elementwise("relu", "maximum (x, 0.0:F32)"),
        elementwise("sigmoid", "%math.slf 0 x"),
        elementwise("swish", "mul (x, %math.slf 0 x)"),
        elementwise("elu", "%core.select (%math.cmp.g 0 (x, 0.0:F32), x, sub (exp x, 1.0:F32))"),
        elementwise("hardshrink", "%core.select (%core.bit2.or_ 2 (%math.cmp.l 0 (x, -0.5:F32), %math.cmp.g 0 (x, 0.5:F32)), x, 0.0:F32)"),
        elementwise("hardswish", "div (mul (x, minimum (maximum (add (x, 3.0:F32), 0.0:F32), 6.0:F32)), 6.0:F32)"),
        elementwise("gelu_tanh", "mul (mul (0.5:F32, x), add (1.0:F32, %math.tri.tanh 0 (mul (0.7978845608:F32, add (x, mul (0.044715:F32, mul (mul (x, x), x)))))))"),
        elementwise("gelu_erf", "mul (mul (0.5:F32, x), add (1.0:F32, %math.er.f 0 (div (x, sqrt (2.0:F32)))))"),
        elementwise("math_pow", "%math.pow 0 (x, y)", 257),
        elementwise("math_erf", "%math.er.f 0 x", 257),
    ]
    for op, expression in [("pow", "%math.pow 0 (x, y)"), ("erf", "%math.er.f 0 x"),
                           ("erfc", "%math.er.fc 0 x")]:
        result.append(Case(f"math_{op}_f64", f"math_{op}", 8193, 8193, 8193, f"""
    let (g, x) = %mem.load (g, %mem.lea (a, %core.conv.u 8193 i));
    let (g, y) = %mem.load (g, %mem.lea (b, %core.conv.u 8193 i));
    return (g, {expression});""", a_type="F64", b_type="F64", out_type="F64"))
    result += [replace(elementwise("add", "add (x, y)", n), name=f"add_n{n}") for n in [1, 256, 257]]
    result += [
        Case("transpose", "transpose", 37 * 65, 1, 37 * 65,
             """let index = offset (imod (i, 37I64), idiv (i, 37I64), 65I64);
    let (g, value) = load 2405 (g, a, index);
    return (g, value);""", 65, 37),
        Case("embedding", "embedding", 23 * 17, 13, 13 * 17,
             """let (g, index) = load 13 (g, b, idiv (i, 17I64));
    let index = %math.conv.f2u 0 index;
    let (g, value) = load 391 (g, a, offset (index, imod (i, 17I64), 17I64));
    return (g, value);""", 17, 23, 13),
    ]
    for name, n, rows in [("reduce_sum", 4099, 1), ("reduce_rows", 513, 19),
                          ("reduce_max", 513, 19), ("dot", 4099, 1), ("sgemv", 259, 37),
                          ("histogram", 4099, 17)]:
        family = "reduce_sum" if name == "reduce_rows" else name
        na = n * rows if name in ["reduce_rows", "reduce_max", "sgemv"] else n
        nb = n if name in ["dot", "sgemv"] else 1
        index = f"offset (i, j, {n}I64)" if name in ["reduce_rows", "reduce_max", "sgemv"] else "j"
        step = f"let (g, x) = load {na} (g, a, {index});"
        if name in ["dot", "sgemv"]:
            step += f"\n        let (g, y) = load {nb} (g, b, j);"
            value = "add (acc, mul (x, y))"
        elif name == "histogram":
            value = "add (acc, %core.select (%math.cmp.e 0 (x, as_float i), 1.0:F32, 0.0:F32))"
        else:
            value = "maximum (acc, x)" if name == "reduce_max" else "add (acc, x)"
        init = "-1.0e30:F32" if name == "reduce_max" else "0.0:F32"
        result.append(Case(name, family, na, nb, rows, f"""
    fun step (g: GlobalM, acc: F32, j: I64)@tt: State =
        {step}
        return (g, {value});
    ret state = fold $ ({n}, step, g, {init});
    return state;""", n, rows, strategy="serial reduction per output; SIMT across outputs"))
    result.append(Case("sgemm_naive", "sgemm", 33 * 17, 17 * 35, 33 * 35, """
    fun step (g: GlobalM, acc: F32, j: I64)@tt: State =
        let (g, x) = load 561 (g, a, offset (idiv (i, 35I64), j, 17I64));
        let (g, y) = load 595 (g, b, offset (j, imod (i, 35I64), 35I64));
        return (g, add (acc, mul (x, y)));
    ret state = fold $ (17, step, g, 0.0:F32);
    return state;""", 35, 33, 17))
    for op in ["softmax", "softmax_safe", "layer_norm", "rms_norm"]:
        w, h = 129, 7
        prefix = f"""
    let row = idiv (i, {w}I64);
    fun sum (g: GlobalM, acc: F32, j: I64)@tt: State =
        let (g, x) = load {w*h} (g, a, offset (row, j, {w}I64));
        return (g, add (acc, x));
    fun max (g: GlobalM, acc: F32, j: I64)@tt: State =
        let (g, x) = load {w*h} (g, a, offset (row, j, {w}I64));
        return (g, maximum (acc, x));
"""
        if op.startswith("softmax"):
            prefix += ("    ret (g, m) = fold $ (129, max, g, -1.0e30:F32);\n" if op.endswith("safe")
                       else "    let m = 0.0:F32;\n")
            body = """
    fun denominator (g: GlobalM, acc: F32, j: I64)@tt: State =
        let (g, x) = load 903 (g, a, offset (row, j, 129I64));
        return (g, add (acc, exp (sub (x, m))));
    ret (g, den) = fold $ (129, denominator, g, 0.0:F32);
    let (g, x) = load 903 (g, a, i);
    return (g, div (exp (sub (x, m)), den));
"""
        else:
            prefix += ("    ret (g, total) = fold $ (129, sum, g, 0.0:F32);\n"
                       "    let mean = div (total, 129.0:F32);\n" if op == "layer_norm"
                       else "    let mean = 0.0:F32;\n")
            body = """
    fun squares (g: GlobalM, acc: F32, j: I64)@tt: State =
        let (g, x) = load 903 (g, a, offset (row, j, 129I64));
        let d = sub (x, mean);
        return (g, add (acc, mul (d, d)));
    ret (g, total) = fold $ (129, squares, g, 0.0:F32);
    let scale = sqrt (add (div (total, 129.0:F32), 0.00001:F32));
    let (g, x) = load 903 (g, a, i);
    return (g, div (sub (x, mean), scale));
"""
        result.append(Case(op, op, w * h, 1, w * h, prefix + body, w, h,
                           strategy="serial row statistics recomputed per output baseline"))
    result.append(Case("softmax_online", "softmax_safe", 903, 1, 903, """
    let row = idiv (i, 129I64);
    %affine.For body done (0I64, 129I64, 1I64, (g, -1.0e30:F32, 0.0:F32))
    where
        con body (j: I64, state: [GlobalM, F32, F32], yield: Cn [GlobalM, F32, F32]) =
            let (g, m, den) = state;
            let (g, x) = load 903 (g, a, offset (row, j, 129I64));
            let next = maximum (m, x);
            yield (g, next, add (mul (den, exp (sub (m, next))), exp (sub (x, next))));
        con done (state: [GlobalM, F32, F32]) =
            let (g, m, den) = state;
            let (g, x) = load 903 (g, a, i);
            return (g, div (exp (sub (x, m)), den));
    end;""", 129, 7, strategy="online max/denominator recurrence per output baseline"))
    result.append(Case("layer_norm_welford", "layer_norm", 903, 1, 903, """
    let row = idiv (i, 129I64);
    %affine.For body done (0I64, 129I64, 1I64, (g, 0.0:F32, 0.0:F32))
    where
        con body (j: I64, state: [GlobalM, F32, F32], yield: Cn [GlobalM, F32, F32]) =
            let (g, mean, m2) = state;
            let (g, x) = load 903 (g, a, offset (row, j, 129I64));
            let delta = sub (x, mean);
            let next = add (mean, div (delta, as_float (%core.wrap.add 0 (j, 1I64))));
            yield (g, next, add (m2, mul (delta, sub (x, next))));
        con done (state: [GlobalM, F32, F32]) =
            let (g, mean, m2) = state;
            let (g, x) = load 903 (g, a, i);
            return (g, div (sub (x, mean), sqrt (add (div (m2, 129.0:F32), 0.00001:F32))));
    end;""", 129, 7, strategy="Welford recurrence per output baseline"))
    result.append(Case("rope", "rope", 31 * 34, 1, 31 * 34, """
    let row = idiv (i, 34I64);
    let pair = idiv (imod (i, 34I64), 2I64);
    let base = offset (row, %core.wrap.mul 0 (pair, 2I64), 34I64);
    let (g, x) = load 1054 (g, a, base);
    let (g, y) = load 1054 (g, a, %core.wrap.add 0 (base, 1I64));
    let power = div (%math.minus 0 (mul (2.0:F32, as_float pair)), 34.0:F32);
    let angle = mul (as_float row, %math.pow 0 (10000.0:F32, power));
    let c = %math.tri.cos 0 angle;
    let s = %math.tri.sin 0 angle;
    return (g, %core.select (%core.icmp.e (imod (i, 2I64), 0I64),
        sub (mul (x, c), mul (y, s)), add (mul (x, s), mul (y, c))));""", 34, 31))
    result.append(Case("merge_attention", "merge_attention", 514, 514, 257, """
    let (g, x) = load 514 (g, a, i);
    let (g, y) = load 514 (g, a, %core.wrap.add 0 (i, 257I64));
    let (g, lx) = load 514 (g, b, i);
    let (g, ly) = load 514 (g, b, %core.wrap.add 0 (i, 257I64));
    let m = maximum (lx, ly);
    let wx = exp (sub (lx, m));
    let wy = exp (sub (ly, m));
    return (g, div (add (mul (x, wx), mul (y, wy)), add (wx, wy)));"""))
    for base in list(result):
        if base.name in ["add", "mul", "relu", "sigmoid", "swish", "elu", "hardshrink", "hardswish", "gelu_tanh", "gelu_erf"]:
            body = base.body.replace(
                "let (g, x) = load 4099 (g, a, i);",
                "let (g, x) = %mem.load (g, %mem.lea (a, %core.conv.u 4099 i));\n"
                "    let x = %math.conv.f2f %math.f32 x;"
            ).replace(
                "let (g, y) = load 4099 (g, b, i);",
                "let (g, y) = %mem.load (g, %mem.lea (b, %core.conv.u 4099 i));\n"
                "    let y = %math.conv.f2f %math.f32 y;"
            ).replace("return (g, ", "return (g, %math.conv.f2f %math.f16 (").removesuffix(");") + "));"
            result.append(replace(base, name=base.name + "_f16", body=body, a_type="F16", b_type="F16",
                                  out_type="F16", strategy="FP16 storage; FP32 operation; FP16 output"))
    result.append(Case("embedding_i32", "embedding", 391, 13, 221, """
    let (g, index) = %mem.load (g, %mem.lea (b, %core.conv.u 13 (idiv (i, 17I64))));
    let index = %core.conv.u 0 index;
    let (g, value) = load 391 (g, a, offset (index, imod (i, 17I64), 17I64));
    return (g, value);""", 17, 23, 13, b_type="I32"))
    result.append(Case("histogram_i32", "histogram", 4099, 1, 17, """
    fun count (g: GlobalM, acc: I32, j: I64)@tt: [GlobalM, I32] =
        let (g, x) = %mem.load (g, %mem.lea (a, %core.conv.u 4099 j));
        let equal = %core.icmp.e (x, %core.conv.u 4294967296 i);
        return (g, %core.wrap.add 0 (acc, %core.select (equal, 1I32, 0I32)));
    ret state = fold $ (4099, count, g, 0I32);
    return state;""", 4099, 17, a_type="I32", out_type="I32",
                       strategy="bin-owned integer counts; no atomics; O(bins * input) baseline"))
    result.append(Case("hgemv_f16_f32", "sgemv", 9583, 259, 37, """
    fun step (g: GlobalM, acc: F32, j: I64)@tt: State =
        let (g, x) = %mem.load (g, %mem.lea (a, %core.conv.u 9583 (offset (i, j, 259I64))));
        let (g, y) = %mem.load (g, %mem.lea (b, %core.conv.u 259 j));
        return (g, add (acc, mul (%math.conv.f2f %math.f32 x, %math.conv.f2f %math.f32 y)));
    ret state = fold $ (259, step, g, 0.0:F32);
    return state;""", 259, 37, a_type="F16", b_type="F16",
                       strategy="SIMT FP16 loads; FP32 multiplication and accumulation"))
    result += attention_cases()
    result += cooperative_cases()
    result += nms_cases()
    for base in list(result):
        if base.name in ["softmax_safe", "softmax_online", "layer_norm", "layer_norm_welford", "rms_norm",
                          "rope", "sgemm_naive", "dot"]:
            body = base.body.replace("load ", "load_half ")
            result.append(replace(base, name=base.name + "_f16_f32", body=body, a_type="F16",
                                  b_type="F16" if base.name in ["sgemm_naive", "dot"] else "F32",
                                  strategy=base.strategy + "; FP16 inputs, FP32 arithmetic/output"))
        if base.name in ["transpose", "transpose_shared"]:
            width, height = 2053, 1031
            body = (base.body.replace("2405", str(width * height)).replace("37I64", f"{height}I64")
                    .replace("65I64", f"{width}I64").replace("5I64", f"{(width+15)//16}I64"))
            result.append(replace(base, name=base.name + "_large", na=width * height, no=width * height,
                                  width=width, height=height, body=body,
                                  groups=((width + 15) // 16) * ((height + 15) // 16) if base.groups else 0))
    return result


def nms_cases():
    body = """
    let box = %core.select (%core.icmp.ul (tid, 37I64), tid, 0I64);
    fun corners (g: GlobalM, box: I64)@tt: [GlobalM, F32, F32, F32, F32] =
        let (g, x1) = load 148 (g, a, offset (box, 0I64, 4I64));
        let (g, y1) = load 148 (g, a, offset (box, 1I64, 4I64));
        let (g, x2) = load 148 (g, a, offset (box, 2I64, 4I64));
        let (g, y2) = load 148 (g, a, offset (box, 3I64, 4I64));
        return (g, x1, y1, x2, y2);
    ret (g, x1, y1, x2, y2) = corners $ (g, box);
    let area = mul (sub (x2, x1), sub (y2, y1));
    fun mask (g: GlobalM, bits: I64, j: I64)@tt: [GlobalM, I64] =
        ret (g, u1, v1, u2, v2) = corners $ (g, j);
        let width = maximum (0.0:F32, sub (minimum (x2, u2), maximum (x1, u1)));
        let height = maximum (0.0:F32, sub (minimum (y2, v2), maximum (y1, v1)));
        let intersection = mul (width, height);
        let other = mul (sub (u2, u1), sub (v2, v1));
        let threshold = mul (0.5:F32, sub (add (area, other), intersection));
        let suppress = %core.bit2.and_ 2 (%core.icmp.ug (j, box), %math.cmp.g 0 (intersection, threshold));
        let bit = %core.select (suppress, %core.wrap.shl 0 (1I64, j), 0I64);
        return (g, %core.bit2.or_ 0 (bits, bit));
    ret (g, bits) = fold @I64 $ (37, mask, g, 0I64);
    let s = %mem.store (s, %mem.lea (p, %core.conv.u 256 tid), bits);
    let (g, s) = %gpu.sync_work_items (g, s);
    %affine.For body done (0I64, 37I64, 1I64, (g, s, 0I64))
    where
        con body (j: I64, state: [GlobalM, %gpu.SharedM, I64], yield: Cn [GlobalM, %gpu.SharedM, I64]) =
            let (g, s, removed) = state;
            let selected = %core.icmp.e (%core.bit2.and_ 0 (removed, %core.wrap.shl 0 (1I64, j)), 0I64);
            let (s, mask) = %mem.load (s, %mem.lea (p, %core.conv.u 256 j));
            yield (g, s, %core.select (selected, %core.bit2.or_ 0 (removed, mask), removed));
        con done (state: [GlobalM, %gpu.SharedM, I64]) =
            let (g, s, removed) = state;
            let kept = %core.icmp.e (%core.bit2.and_ 0 (removed, %core.wrap.shl 0 (1I64, box)), 0I64);
"""
    body += "        " + block_store(37, "%core.icmp.ul (tid, 37I64)", "tid",
                                     "%core.select (kept, 1.0:F32, 0.0:F32)").replace("\n", "\n        ")
    body += "\n    end;"
    return [Case("nms_presorted", "nms_presorted", 148, 37, 37, body, 4, 37,
                 strategy="N<=64 presorted boxes; per-box parallel IoU masks; replicated greedy resolution; keep flags",
                 groups=1, shared_type="I64")]


def attention_cases():
    result = []
    for online, causal in [(False, False), (True, False), (True, True)]:
        body = """
    let row = idiv (i, 9I64);
    let col = imod (i, 9I64);
    fun score (g: GlobalM, key: I64)@tt: State =
        fun dot (g: GlobalM, acc: F32, j: I64)@tt: State =
            let (g, q) = load 153 (g, a, offset (row, j, 9I64));
            let (g, k) = load 306 (g, b, offset (key, j, 9I64));
            return (g, add (acc, mul (q, k)));
        ret (g, value) = fold $ (9, dot, g, 0.0:F32);
        let value = div (value, 3.0:F32);
"""
        body += ("        let value = %core.select (%core.icmp.ule (key, row), value, -1.0e30:F32);\n" if causal else "")
        body += "        return (g, value);\n"
        if online:
            body += """
    %affine.For body done (0I64, 17I64, 1I64, (g, -1.0e30:F32, 0.0:F32, 0.0:F32))
    where
        con body (key: I64, state: [GlobalM, F32, F32, F32], yield: Cn [GlobalM, F32, F32, F32]) =
            let (g, m, den, numerator) = state;
            ret (g, x) = score $ (g, key);
            let next = maximum (m, x);
            let scale = exp (sub (m, next));
            let weight = exp (sub (x, next));
            let (g, v) = load 306 (g, b, %core.wrap.add 0 (153I64, offset (key, col, 9I64)));
            yield (g, next, add (mul (den, scale), weight), add (mul (numerator, scale), mul (weight, v)));
        con done (state: [GlobalM, F32, F32, F32]) =
            let (g, _, den, numerator) = state;
            return (g, div (numerator, den));
    end;"""
        else:
            body += """
    fun max (g: GlobalM, acc: F32, key: I64)@tt: State =
        ret (g, x) = score $ (g, key);
        return (g, maximum (acc, x));
    ret (g, m) = fold $ (17, max, g, -1.0e30:F32);
    %affine.For body done (0I64, 17I64, 1I64, (g, 0.0:F32, 0.0:F32))
    where
        con body (key: I64, state: [GlobalM, F32, F32], yield: Cn [GlobalM, F32, F32]) =
            let (g, den, numerator) = state;
            ret (g, x) = score $ (g, key);
            let weight = exp (sub (x, m));
            let (g, v) = load 306 (g, b, %core.wrap.add 0 (153I64, offset (key, col, 9I64)));
            yield (g, add (den, weight), add (numerator, mul (weight, v)));
        con done (state: [GlobalM, F32, F32]) =
            let (g, den, numerator) = state;
            return (g, div (numerator, den));
    end;"""
        name = "attention_causal" if causal else "attention_online" if online else "attention_naive"
        result.append(Case(name, "attention", 153, 306, 153, body, 9, 17, int(causal),
                           strategy="SIMT per-output attention; online recurrence" if online
                           else "SIMT per-output attention; stable two-pass softmax"))
    return result


def block_store(no, condition, index, value):
    return f"""
    (skip, write)#({condition}) (g, s, c, l)
    where
        con skip (g: GlobalM, s: %gpu.SharedM, c: %gpu.ConstM, l: %gpu.LocalM) = return (g, s, c, l);
        con write (g: GlobalM, s: %gpu.SharedM, c: %gpu.ConstM, l: %gpu.LocalM) =
            let g = %mem.store (g, %mem.lea (out, %core.conv.u {no} ({index})), {value});
            return (g, s, c, l);
    end;"""


def cooperative_cases():
    result = []
    for name, family, n, rows in [
        ("reduce_sum_block", "reduce_sum", 4099, 1), ("reduce_rows_block", "reduce_sum", 513, 19),
        ("reduce_max_block", "reduce_max", 513, 19), ("dot_block", "dot", 4099, 1),
        ("sgemv_block", "sgemv", 259, 37),
    ]:
        na, nb = n * rows, n if family in ["dot", "sgemv"] else 1
        combine = "maximum" if family == "reduce_max" else "add"
        initial = "-1.0e30:F32" if family == "reduce_max" else "0.0:F32"
        body = f"""
    fun partial (g: GlobalM, acc: F32, j: I64)@tt: State =
        let col = offset (j, tid, 256I64);
        let valid = %core.icmp.ul (col, {n}I64);
        let col = %core.select (valid, col, 0I64);
        let (g, x) = load {na} (g, a, offset (gid, col, {n}I64));
"""
        if family in ["dot", "sgemv"]:
            body += f"        let (g, y) = load {nb} (g, b, col);\n        let x = mul (x, y);\n"
        body += f"""
        let x = %core.select (valid, x, {initial});
        return (g, {combine} (acc, x));
    ret (g, value) = fold $ ({(n + 255) // 256}, partial, g, {initial});
    ret (g, s, total) = block_reduce $ (g, s, p, tid, value, {combine});
"""
        body += block_store(rows, "%core.icmp.e (tid, 0I64)", "gid", "total")
        result.append(Case(name, family, na, nb, rows, body, n, rows,
                           strategy="256-thread shared-memory tree; strided thread partials", groups=rows))
    for op in ["softmax_safe", "log_softmax", "layer_norm", "rms_norm"]:
        body = """
    let valid = %core.icmp.ul (tid, 129I64);
    let col = %core.select (valid, tid, 0I64);
    let (g, x) = load 903 (g, a, offset (gid, col, 129I64));
"""
        if "softmax" in op:
            body += """
    ret (g, s, m) = block_reduce $ (g, s, p, tid, %core.select (valid, x, -1.0e30:F32), maximum);
    let numerator = exp (sub (x, m));
    ret (g, s, total) = block_reduce $ (g, s, p, tid, %core.select (valid, numerator, 0.0:F32), add);
"""
            value = "sub (sub (x, m), %math.exp.log 0 total)" if op == "log_softmax" else "div (numerator, total)"
        else:
            if op == "layer_norm":
                body += """
    ret (g, s, total) = block_reduce $ (g, s, p, tid, %core.select (valid, x, 0.0:F32), add);
    let mean = div (total, 129.0:F32);
"""
            else:
                body += "    let mean = 0.0:F32;\n"
            body += """
    let d = sub (x, mean);
    ret (g, s, total) = block_reduce $ (g, s, p, tid, %core.select (valid, mul (d, d), 0.0:F32), add);
    let scale = sqrt (add (div (total, 129.0:F32), 0.00001:F32));
"""
            value = "div (d, scale)"
        body += block_store(903, "valid", "offset (gid, tid, 129I64)", value)
        result.append(Case(op + "_block", op, 903, 1, 903, body, 129, 7,
                           strategy="one 256-thread block per row; shared-memory trees", groups=7))
    result.append(Case("transpose_shared", "transpose", 2405, 1, 2405, """
    let tx = imod (tid, 16I64);
    let ty = idiv (tid, 16I64);
    let bx = imod (gid, 5I64);
    let by = idiv (gid, 5I64);
    let row = offset (by, ty, 16I64);
    let col = offset (bx, tx, 16I64);
    let valid = %core.bit2.and_ 2 (%core.icmp.ul (row, 37I64), %core.icmp.ul (col, 65I64));
    let index = %core.select (valid, offset (row, col, 65I64), 0I64);
    let (g, x) = load 2405 (g, a, index);
    let s = %mem.store (s, %mem.lea (p, %core.conv.u 272 (offset (ty, tx, 17I64))), x);
    let (g, s) = %gpu.sync_work_items (g, s);
    let (s, x) = %mem.load (s, %mem.lea (p, %core.conv.u 272 (offset (tx, ty, 17I64))));
    let row = offset (bx, ty, 16I64);
    let col = offset (by, tx, 16I64);
    let valid = %core.bit2.and_ 2 (%core.icmp.ul (row, 65I64), %core.icmp.ul (col, 37I64));
""" + block_store(2405, "valid", "offset (row, col, 37I64)", "x"), 65, 37,
                       strategy="16x16 shared-memory transpose with stride 17 and fringe guards", groups=15, shared_n=272))
    return result


def emit(case, directory):
    source = HERE.joinpath("simt.mim").read_text()
    if "F64" in [case.a_type, case.b_type, case.out_type]:
        source += "\nlet F64 = %math.F64;\n"
    body = case.body.replace("fold $", "fold @I32 $" if case.name == "histogram_i32" else "fold @F32 $")
    if case.groups:
        source += "\n" + HERE.joinpath("block.mim").read_text()
        source += f"""
fun evaluate (g: GlobalM, s: %gpu.SharedM, c: %gpu.ConstM, l: %gpu.LocalM, gid tid: I64,
              p: %gpu.SharedPtr «{case.shared_n}; {case.shared_type}»,
              a: %gpu.GlobalPtr «{case.na}; F32», b: %gpu.GlobalPtr «{case.nb}; F32»,
              out: %gpu.GlobalPtr «{case.no}; F32»)@tt: [GlobalM, %gpu.SharedM, %gpu.ConstM, %gpu.LocalM] =
    {body}
"""
        runner = f"run_block @{case.shared_type} $ ({case.na}, {case.nb}, {case.no}, {case.groups}, {case.shared_n}, evaluate, m, a, b, out)"
    else:
        source += f"""

fun evaluate (g: GlobalM, a: %gpu.GlobalPtr «{case.na}; {case.a_type}»,
              b: %gpu.GlobalPtr «{case.nb}; {case.b_type}», i: I64)@tt: [GlobalM, {case.out_type}] =
    {body}
"""
        runner = f"run @({case.a_type}, {case.b_type}, {case.out_type}) $ ({case.na}, {case.nb}, {case.no}, evaluate, m, a, b, out)"
    source += f"""

fun extern execute (m: %mem.M 0, a: %mem.Ptr0 «{case.na}; {case.a_type}»,
                    b: %mem.Ptr0 «{case.nb}; {case.b_type}», out: %mem.Ptr0 «{case.no}; {case.out_type}»): %mem.M 0 =
    ret m = {runner};
    return m;
"""
    path = directory / f"{case.name}.mim"
    path.write_text(source)
    return path


def logged(command, log, cwd):
    with log.open("w") as output:
        result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode < 0:
            output.write(f"\nProcess terminated by signal {-result.returncode}\n")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="")
    parser.add_argument("--sanitize", choices=["memcheck", "racecheck", "synccheck", "initcheck"])
    parser.add_argument("--emit-only", action="store_true")
    parser.add_argument("--math-runtime", type=Path, default=HERE / "artifacts/math-toolchain")
    parser.add_argument("--no-software-math", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    parser.add_argument("--mim", type=Path, default=ROOT / "build-clang/bin/mim")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = [case for case in cases() if not args.filter or re.search(args.filter, case.name)]
    if not chosen:
        parser.error("No cases match --filter")
    if args.emit_only:
        for case in chosen:
            directory = args.out / case.name
            directory.mkdir(exist_ok=True)
            print(emit(case, directory))
        return
    math_flags = []
    if not args.no_software_math:
        args.math_runtime = args.math_runtime.resolve()
        for name in ["runtime.bc", "lower_math.so", "manifest.json"]:
            if not (args.math_runtime / name).is_file():
                raise SystemExit("Build software math first: python3 bench/leetcuda-simt/toolchain/build.py")
        math_flags = [
            f"-Xll_nvptx:Xlink_llvm={args.math_runtime / 'runtime.bc'}",
            f'-Xll_nvptx:Xopt=-load-pass-plugin={args.math_runtime / "lower_math.so"} '
            '-passes="default<O2>,mim-nvptx-math,cgscc(inline),globaldce,nvvm-reflect"',
        ]
    harness = args.out / "check"
    if logged(["nvcc", "-std=c++17", "-O2", "-arch=sm_75", str(HERE / "check.cu"), "-lcuda", "-o", str(harness)],
              args.out / "harness-build.log", HERE):
        raise SystemExit("Harness build failed; see harness-build.log")
    manifest = args.out / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else []
    records = {record["name"]: record for record in previous}
    selected = []
    for case in chosen:
        directory = args.out / case.name
        directory.mkdir(exist_ok=True)
        source = emit(case, directory)
        start = time.monotonic()
        command = [
            str(args.mim.resolve()), str(source), "-p", "opt", "-p", "ll_nvptx",
            "-Xll_nvptx:sm=75", "-Xll_nvptx:embed",
            *math_flags,
        ]
        record = asdict(case)
        record.pop("body")
        record["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        record["harness_sha256"] = hashlib.sha256((HERE / "check.cu").read_bytes()).hexdigest()
        record["compile_seconds"] = 0
        code = logged(command, directory / "compile.log", directory)
        record["compile_seconds"] = time.monotonic() - start
        record["compile_returncode"] = code
        record["status"] = "compile_failed" if code else "compiled"
        if not code:
            record["cubin_sha256"] = hashlib.sha256((directory / f"{case.name}_dev.cubin").read_bytes()).hexdigest()
            previous_record = records.get(case.name, {})
            same_inputs = all(record[key] == previous_record.get(key)
                              for key in ["source_sha256", "harness_sha256", "cubin_sha256"])
            record["sanitizers"] = dict(previous_record.get("sanitizers", {})) if same_inputs else {}
            llvm = (directory / f"{case.name}_dev.ll").read_text()
            kernels = re.findall(r"^define ptx_kernel void @([^(]+)\(", llvm, re.MULTILINE)
            if len(kernels) != 1:
                raise RuntimeError(f"{case.name}: expected one kernel, got {kernels}")
            record["kernel"] = kernels[0]
            check = [
                str(harness), str(directory / f"{case.name}_dev.cubin"), kernels[0], case.family,
                *map(str, [case.na, case.nb, case.no, case.width, case.height, case.depth,
                           case.groups or (case.no + 255) // 256]),
                case.a_type, case.b_type, case.out_type,
            ]
            code = logged(check, directory / "check.log", directory)
            record["status"] = "check_failed" if code else "passed"
            if not code:
                record.update(json.loads((directory / "check.log").read_text()))
                if args.sanitize:
                    sanitizer = [
                        "compute-sanitizer", "--tool", args.sanitize, "--error-exitcode", "99",
                        "--log-file", str(directory / f"{args.sanitize}.log"), *check, "--check-only",
                    ]
                    code = logged(sanitizer, directory / f"{args.sanitize}-check.log", directory)
                    report = (directory / f"{args.sanitize}.log").read_text()
                    clean = re.search(r"(ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards)", report)
                    if not clean:
                        code = code or 1
                    record["sanitizers"][args.sanitize] = "failed" if code else "passed"
                    if code:
                        record["status"] = "sanitizer_failed"
        records[case.name] = record
        selected.append(record)
        manifest.write_text(json.dumps(list(records.values()), indent=2) + "\n")
        print(f'{case.name}: {record["status"]}', flush=True)
    inputs = [HERE / name for name in ["run.py", "simt.mim", "block.mim", "check.cu"]]
    inputs += [ROOT / "src/mim/plug/ll_nvptx/phase/ll_nvptx.cpp",
               ROOT / "src/mim/plug/ll_nvptx/ll_nvptx.cpp",
               ROOT / "src/mim/plug/affine/phase/lower_for.cpp",
               ROOT / "src/mim/plug/mem/phase/seo.cpp"]
    inputs += [HERE / "toolchain" / name for name in ["sources.json", "build.py", "lower_math.cpp", "pow.ll",
                                                    "pow-subnormal.patch"]]
    result = {
        "device": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
        "compiler_flags": ["-p", "opt", "-p", "ll_nvptx", "-Xll_nvptx:sm=75", "-Xll_nvptx:embed", *math_flags],
        "math_runtime": None if args.no_software_math else json.loads(
            (args.math_runtime / "manifest.json").read_text()
        ),
        "method": "three fixtures; full-output CPU reference (long double for F64 math); 20 warmups; median of 5 x 100 CUDA-event launches; kernel only",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "cases": list(records.values()),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if any(r["status"].endswith("failed") for r in selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
