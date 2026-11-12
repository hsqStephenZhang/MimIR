"""flash_tc.py BH L [BC]: writes a hand-written FA2-style TF32 attention kernel in Mim for BH heads of L × 64.

A work-group takes 64 query rows, a warp 16 of them; K and V move through shared memory in blocks of 64 keys,
double-buffered with cp.async. S and P stay in the warps' registers: within each 8 keys, the product's accumulator
holds keys 2t and 2t+1 where the next product's A fragment wants keys t and t+4, so the keys are permuted to match and
V is read in the same order. Row maxima are reduced over the four lanes of a quad with shfl.xor; the sums are reduced
once at the end.
"""
import math
import sys

BH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
D = 64
BR = 64
BC = int(sys.argv[3]) if len(sys.argv) > 3 else 64
QB = L // BR
G = BH * QB
KB = L // BC
SCALE = 1 / math.sqrt(D) * math.log2(math.e)


def i64(x):
    return f"{x}I64"


out = []
p = out.append
p(f"""// FA2-style TF32 attention, {BH} heads of {L} × {D}: 64 query rows per work-group, 16 per warp, K and V in blocks of
// {BC} keys through double-buffered shared memory; S and P stay in registers.
plugin core;
plugin gpu;
plugin mem;
plugin math;
plugin affine;

use core.ops.u.w;

let F32  = math.F32;
let fast = math.mode.fast;
let T    = «{BH}, {L}, {D}; F32»;
let Frag = «8; «4; F32»»;

lam ix (n: Nat) (x: I64): Idx n = core.conv.u n x;
lam shr (x: I64, s: I64): I64 = core.shr.l (x, s);
lam band (x: I64, s: I64): I64 = core.bit2.and 0 (x, s);
lam fmax (x y: F32): F32 = math.extrema.fmax fast (x, y);
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
    let (m, sk) = gpu.slot_shared «2, {BC}, 17, 4; F32» m;
    let (m, sv) = gpu.slot_shared «2, {BC}, 17, 4; F32» m;
    let ks      = core.bitcast (mem.Ptr0 «2, {BC}, 68; F32») sk;
    let vs      = core.bitcast (mem.Ptr0 «2, {BC}, 68; F32») sv;
    let qv      = core.bitcast (mem.Ptr0 T) q;
    let kv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 4}, 4; F32») k;
    let vv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 4}, 4; F32») v;
    let ov      = core.bitcast (mem.Ptr0 T) o;
    let tid     = core.conv.u 0 item_id;
    let gid     = core.conv.u 0 group_id;
    let h       = shr (gid, {i64(QB.bit_length() - 1)});
    let row0    = band (gid, {i64(QB - 1)}) * {i64(BR)} + shr (tid, 5I64) * 16I64;
    let lane    = band (tid, 31I64);
    let g       = shr (lane, 2I64);
    let t       = band (lane, 3I64);
    lam at (p: mem.Ptr0 T, r c: I64): mem.Ptr0 F32 = mem.lea (mem.lea (mem.lea (p, ix {BH} h), ix {L} r), ix {D} c);
    lam fill (m: mem.M 0, buf: Idx 2, kb: I64): mem.M 0 =""")
for j in range(BC * D // 4 // 128):
    p(f"""        let c{j}       = tid + {i64(128 * j)};
        let r{j}       = kb * {i64(BC)} + shr (c{j}, 4I64);
        let (m, a{j})  = mem.load (m, mem.lea (mem.lea (mem.lea (kv, ix {BH} h), ix {L} r{j}), ix {D // 4} (band (c{j}, 15I64))));
        let m         = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sk, buf), ix {BC} (shr (c{j}, 4I64))), ix 17 (band (c{j}, 15I64))), a{j});
        let (m, b{j})  = mem.load (m, mem.lea (mem.lea (mem.lea (vv, ix {BH} h), ix {L} r{j}), ix {D // 4} (band (c{j}, 15I64))));
        let m         = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sv, buf), ix {BC} (shr (c{j}, 4I64))), ix 17 (band (c{j}, 15I64))), b{j});""")
p("        m;")
# Q fragments, scaled by 1/sqrt(D)·log2(e) so that the softmax can use exp2.
for s in range(D // 8):
    for e, (dr, dc) in enumerate([(0, 0), (8, 0), (0, 4), (8, 4)]):
        p(f"""    let (m, q{s}{e}) = mem.load (m, at (qv, row0 + g + {i64(dr)}, {i64(s * 8 + dc)} + t));""")
    p(f"""    let qa{s} = ({", ".join(f"math.arith.mul fast (q{s}{e}, {SCALE:.9f}:F32)" for e in range(4))});""")
p(f"""    lam step (m: mem.M 0, buf: Idx 2, o: Frag, mx0 mx1 l0 l1: F32): [mem.M 0, Frag, F32, F32, F32, F32] =""")
for n in range(BC // 8):
    p(f"""        let s{n} = ‹4; 0.0:F32›;""")
    for s in range(D // 8):
        p(f"""        let (m, kb{n}{s}0) = mem.load (m, mem.lea (mem.lea (mem.lea (ks, buf), ix {BC} ({i64(n * 8)} + g)), ix 68 ({i64(s * 8)} + t)));
        let (m, kb{n}{s}1) = mem.load (m, mem.lea (mem.lea (mem.lea (ks, buf), ix {BC} ({i64(n * 8)} + g)), ix 68 ({i64(s * 8 + 4)} + t)));
        let s{n} = gpu.mma_tf32 (qa{s}, (kb{n}{s}0, kb{n}{s}1), s{n});""")


def fold(op, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = f"{op} ({acc}, {x})"
    return acc


p(f"""        let r0  = quad_max ({fold("fmax", [f"s{n}#{e}₄" for n in range(BC // 8) for e in (0, 1)])});
        let r1  = quad_max ({fold("fmax", [f"s{n}#{e}₄" for n in range(BC // 8) for e in (2, 3)])});
        let n0  = fmax (mx0, r0);
        let n1  = fmax (mx1, r1);
        let al0 = math.exp.exp2 fast (math.arith.sub fast (mx0, n0));
        let al1 = math.exp.exp2 fast (math.arith.sub fast (mx1, n1));""")
for n in range(BC // 8):
    p(f"""        let p{n} = (math.exp.exp2 fast (math.arith.sub fast (s{n}#0₄, n0)), math.exp.exp2 fast (math.arith.sub fast (s{n}#1₄, n0)),
                  math.exp.exp2 fast (math.arith.sub fast (s{n}#2₄, n1)), math.exp.exp2 fast (math.arith.sub fast (s{n}#3₄, n1)));""")
add = "math.arith.add fast"
p(f"""        let l0 = {add} (math.arith.mul fast (l0, al0), {fold(add, [f"p{n}#{e}₄" for n in range(BC // 8) for e in (0, 1)])});
        let l1 = {add} (math.arith.mul fast (l1, al1), {fold(add, [f"p{n}#{e}₄" for n in range(BC // 8) for e in (2, 3)])});
        let o  = ‹i: 8; (math.arith.mul fast (o#i#0₄, al0), math.arith.mul fast (o#i#1₄, al0),
                         math.arith.mul fast (o#i#2₄, al1), math.arith.mul fast (o#i#3₄, al1))›;""")
for n in range(D // 8):
    p(f"""        let o{n} = o#{n}₈;""")
for j in range(BC // 8):
    p(f"""        let pa{j} = (p{j}#0₄, p{j}#2₄, p{j}#1₄, p{j}#3₄);""")
    for n in range(D // 8):
        p(f"""        let (m, vb{j}{n}0) = mem.load (m, mem.lea (mem.lea (mem.lea (vs, buf), ix {BC} ({i64(j * 8)} + 2I64 * t)), ix 68 ({i64(n * 8)} + g)));
        let (m, vb{j}{n}1) = mem.load (m, mem.lea (mem.lea (mem.lea (vs, buf), ix {BC} ({i64(j * 8 + 1)} + 2I64 * t)), ix 68 ({i64(n * 8)} + g)));
        let o{n} = gpu.mma_tf32 (pa{j}, (vb{j}{n}0, vb{j}{n}1), o{n});""")
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
    for e, (dr, dc, inv) in enumerate([(0, 0, "inv0"), (0, 1, "inv0"), (8, 0, "inv1"), (8, 1, "inv1")]):
        p(f"""            let m = mem.store (m, at (ov, row0 + g + {i64(dr)}, {i64(n * 8 + dc)} + 2I64 * t), math.arith.mul fast (o#{n}₈#{e}₄, {inv}));""")
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
