"""flash_h16.py BH L [f16|bf16]: writes a hand-written FA2-style 16-bit attention kernel in Mim for BH heads of L × 64.

The structure is flash_tc.py's with m16n8k16 products: 64 query rows per work-group, 16 per warp, K and V in blocks of
64 keys double-buffered with cp.async. K's fragments come from ldmatrix and V's from ldmatrix.trans; two accumulator
tiles of S are exactly one A fragment of P, so no key permutation is needed. Softmax runs in F32.
"""
import math
import sys

BH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
TY = sys.argv[3] if len(sys.argv) > 3 else "f16"
TM = {"f16": "math.F16", "bf16": "math.BF16"}[TY]
D = 64
BR = 64
BC = 64
QB = L // BR
G = BH * QB
KB = L // BC
SCALE = 1 / math.sqrt(D) * math.log2(math.e)
NT = BC // 8


def i64(x):
    return f"{x}I64"


def fold(op, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = f"{op} ({acc}, {x})"
    return acc


out = []
p = out.append
p(f"""// FA2-style {TY.upper()} attention with F32 softmax, {BH} heads of {L} × {D}: 64 query rows per work-group, 16 per warp,
// K and V in blocks of {BC} keys through double-buffered shared memory; S and P stay in registers.
plugin core;
plugin gpu;
plugin mem;
plugin math;
plugin affine;

use core.ops.u.w;

let H    = {TM};
let F32  = math.F32;
let fast = math.mode.fast;
let T    = «{BH}, {L}, {D}; H»;
let Frag = «8; «4; F32»»;

lam ix (n: Nat) (x: I64): Idx n = core.conv.u n x;
lam shr (x: I64, s: I64): I64 = core.shr.l (x, s);
lam band (x: I64, s: I64): I64 = core.bit2.and 0 (x, s);
lam fmax (x y: F32): F32 = math.extrema.fmax fast (x, y);
lam narrow (x: F32): H = math.conv.f2f math.{TY} x;
lam quad_max (x: F32): F32 =
    let x = fmax (x, gpu.shfl_xor (x, 1I32));
    fmax (x, gpu.shfl_xor (x, 2I32));
lam quad_sum (x: F32): F32 =
    let x = math.arith.add fast (x, gpu.shfl_xor (x, 1I32));
    math.arith.add fast (x, gpu.shfl_xor (x, 2I32));
lam sync (m: mem.M 0, m3: gpu.SharedM): mem.M 0 =
    let (m1, _) = gpu.sync_work_items (gpu.global_mem m, m3);
    gpu.generic_mem m1;

con kernel (m1: gpu.GlobalM, m3: gpu.SharedM, m4: gpu.ConstM, m5: gpu.LocalM,
            group_id: Idx {G}, item_id: Idx 128, (), (q k v o: gpu.GlobalPtr T),
            return: Cn [gpu.GlobalM, gpu.SharedM, gpu.ConstM, gpu.LocalM]) =
    let m       = gpu.generic_mem m1;
    let (m, sk) = gpu.slot_shared «2, {BC}, 9, 8; H» m;
    let (m, sv) = gpu.slot_shared «2, {BC}, 9, 8; H» m;
    let qv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 2}, 2; H») q;
    let kv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 8}, 8; H») k;
    let vv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 8}, 8; H») v;
    let ov      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 2}, 2; H») o;
    let tid     = core.conv.u 0 item_id;
    let gid     = core.conv.u 0 group_id;
    let h       = shr (gid, {i64(QB.bit_length() - 1)});
    let row0    = band (gid, {i64(QB - 1)}) * {i64(BR)} + shr (tid, 5I64) * 16I64;
    let lane    = band (tid, 31I64);
    let g       = shr (lane, 2I64);
    let t       = band (lane, 3I64);
    lam pair (p: mem.Ptr0 «{BH}, {L}, {D // 2}, 2; H», r c: I64): mem.Ptr0 «2; H» = mem.lea (mem.lea (mem.lea (p, ix {BH} h), ix {L} r), ix {D // 2} c);
    lam fill (m: mem.M 0, buf: Idx 2, kb: I64): mem.M 0 =""")
for j in range(BC * D // 8 // 128):
    p(f"""        let c{j}       = tid + {i64(128 * j)};
        let r{j}       = kb * {i64(BC)} + shr (c{j}, 3I64);
        let (m, a{j})  = mem.load (m, mem.lea (mem.lea (mem.lea (kv, ix {BH} h), ix {L} r{j}), ix {D // 8} (band (c{j}, 7I64))));
        let m         = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sk, buf), ix {BC} (shr (c{j}, 3I64))), ix 9 (band (c{j}, 7I64))), a{j});
        let (m, b{j})  = mem.load (m, mem.lea (mem.lea (mem.lea (vv, ix {BH} h), ix {L} r{j}), ix {D // 8} (band (c{j}, 7I64))));
        let m         = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sv, buf), ix {BC} (shr (c{j}, 3I64))), ix 9 (band (c{j}, 7I64))), b{j});""")
p("        m;")
for s in range(D // 16):
    for r in range(4):
        p(f"""    let (m, q{s}{r}) = mem.load (m, pair (qv, row0 + g + {i64(8 * (r % 2))}, {i64(8 * s + 4 * (r // 2))} + t));""")
    p(f"""    let qa{s} = (q{s}0, q{s}1, q{s}2, q{s}3);""")
p(f"""    lam step (m: mem.M 0, buf: Idx 2, o: Frag, mx0 mx1 l0 l1: F32): [mem.M 0, Frag, F32, F32, F32, F32] =""")
for n in range(NT):
    p(f"""        let s{n} = ‹4; 0.0:F32›;""")
for nn in range(NT // 2):
    for s in range(D // 16):
        p(f"""        let (m, kf{nn}{s}) = gpu.ldmatrix_b16 (4, ff, m, mem.lea (mem.lea (mem.lea (sk, buf), ix {BC} ({i64(16 * nn)} + shr (lane, 4I64) * 8I64 + band (lane, 7I64))), ix 9 ({i64(2 * s)} + band (shr (lane, 3I64), 1I64))));
        let s{2 * nn} = gpu.mma_{TY} (qa{s}, (kf{nn}{s}#0₄, kf{nn}{s}#1₄), s{2 * nn});
        let s{2 * nn + 1} = gpu.mma_{TY} (qa{s}, (kf{nn}{s}#2₄, kf{nn}{s}#3₄), s{2 * nn + 1});""")
for n in range(NT):
    p(f"""        let s{n} = ‹i: 4; math.arith.mul fast (s{n}#i, {SCALE:.9f}:F32)›;""")
p(f"""        let r0  = quad_max ({fold("fmax", [f"s{n}#{e}₄" for n in range(NT) for e in (0, 1)])});
        let r1  = quad_max ({fold("fmax", [f"s{n}#{e}₄" for n in range(NT) for e in (2, 3)])});
        let n0  = fmax (mx0, r0);
        let n1  = fmax (mx1, r1);
        let al0 = math.exp.exp2 fast (math.arith.sub fast (mx0, n0));
        let al1 = math.exp.exp2 fast (math.arith.sub fast (mx1, n1));""")
for n in range(NT):
    p(f"""        let p{n} = (math.exp.exp2 fast (math.arith.sub fast (s{n}#0₄, n0)), math.exp.exp2 fast (math.arith.sub fast (s{n}#1₄, n0)),
                  math.exp.exp2 fast (math.arith.sub fast (s{n}#2₄, n1)), math.exp.exp2 fast (math.arith.sub fast (s{n}#3₄, n1)));""")
add = "math.arith.add fast"
p(f"""        let l0 = {add} (math.arith.mul fast (l0, al0), {fold(add, [f"p{n}#{e}₄" for n in range(NT) for e in (0, 1)])});
        let l1 = {add} (math.arith.mul fast (l1, al1), {fold(add, [f"p{n}#{e}₄" for n in range(NT) for e in (2, 3)])});
        let o  = ‹i: 8; (math.arith.mul fast (o#i#0₄, al0), math.arith.mul fast (o#i#1₄, al0),
                         math.arith.mul fast (o#i#2₄, al1), math.arith.mul fast (o#i#3₄, al1))›;""")
for n in range(D // 8):
    p(f"""        let o{n} = o#{n}₈;""")
for kk in range(BC // 16):
    regs = ", ".join(f"(narrow (p{2 * kk + r // 2}#{2 * (r % 2)}₄), narrow (p{2 * kk + r // 2}#{2 * (r % 2) + 1}₄))" for r in range(4))
    p(f"""        let pa{kk} = ({regs});""")
    for nj in range(D // 16):
        p(f"""        let (m, vf{kk}{nj}) = gpu.ldmatrix_b16 (4, tt, m, mem.lea (mem.lea (mem.lea (sv, buf), ix {BC} ({i64(16 * kk)} + band (lane, 15I64))), ix 9 ({i64(2 * nj)} + shr (lane, 4I64))));
        let o{2 * nj} = gpu.mma_{TY} (pa{kk}, (vf{kk}{nj}#0₄, vf{kk}{nj}#1₄), o{2 * nj});
        let o{2 * nj + 1} = gpu.mma_{TY} (pa{kk}, (vf{kk}{nj}#2₄, vf{kk}{nj}#3₄), o{2 * nj + 1});""")
p(f"""        (m, ({", ".join(f"o{n}" for n in range(D // 8))}), n0, n1, l0, l1);
    let m = fill (m, 0₂, 0I64);
    affine.For block block_exit (0I64, {i64(KB)}, 1I64, (m, ‹8; ‹4; 0.0:F32››, math.neg_inf math.f32, math.neg_inf math.f32, 0.0:F32, 0.0:F32))
    where
        con block (kb: I64, state: [mem.M 0, Frag, F32, F32, F32, F32], continue: Cn [[mem.M 0, Frag, F32, F32, F32, F32]]) =
            let (m, o, mx0, mx1, l0, l1) = state;
            let m    = sync (gpu.async_wait m, m3);
            let buf  = ix 2 (band (kb, 1I64));
            let next = core.select (core.icmp.ul (kb + 1I64, {i64(KB)}), kb + 1I64, {i64(KB - 1)});
            let m    = fill (m, ix 2 (band (kb + 1I64, 1I64)), next);
            continue (step (m, buf, o, mx0, mx1, l0, l1));
        con block_exit (state: [mem.M 0, Frag, F32, F32, F32, F32]) =
            let (m, o, _, _, l0, l1) = state;
            let m  = gpu.async_wait m;
            let inv0 = math.arith.div fast (1.0:F32, quad_sum l0);
            let inv1 = math.arith.div fast (1.0:F32, quad_sum l1);""")
for n in range(D // 8):
    for hh, inv in enumerate(["inv0", "inv1"]):
        p(f"""            let m = mem.store (m, pair (ov, row0 + g + {i64(8 * hh)}, {i64(4 * n)} + t), (narrow (math.arith.mul fast (o#{n}₈#{2 * hh}₄, {inv})), narrow (math.arith.mul fast (o#{n}₈#{2 * hh + 1}₄, {inv}))));""")
p("""            return (gpu.global_mem m, m3, m4, m5);
    end;
""")
p(f"""extern fun flash (m0: mem.M 0, q k v o: mem.Ptr0 T): [mem.M 0, I32] =
    let (m0, m1, m4) = gpu.init m0;
    let (m1, dq)     = mem.alloc (T, 1) m1;
    let (m1, dk)     = mem.alloc (T, 1) m1;
    let (m1, dv)     = mem.alloc (T, 1) m1;
    let (m1, do)     = mem.alloc (T, 1) m1;
    let (m0, m1)     = gpu.copy_to_device.block (m0, m1, q, dq);
    let (m0, m1)     = gpu.copy_to_device.block (m0, m1, k, dk);
    let (m0, m1)     = gpu.copy_to_device.block (m0, m1, v, dv);
    ret (m0, m1, m4) = gpu.launch ({G}, 128, gpu.default_stream, ff, ()) kernel (dq, dk, dv, do) $ (m0, m1, m4);
    let (m0, m1)     = gpu.copy_to_host.block (m0, m1, do, o);
    let m1           = mem.free (m1, dq);
    let m1           = mem.free (m1, dk);
    let m1           = mem.free (m1, dv);
    let m1           = mem.free (m1, do);
    let m0           = gpu.deinit (m0, m1, m4);
    return (m0, 0I32);""")
print("\n".join(out))
