"""flash_wg.py BH L [f16|bf16] [BC]: writes a hand-written 16-bit attention kernel on warpgroup MMAs in Mim for BH heads of L × 64.

Two warpgroups take 64 query rows each; Q, K and V live in 128-byte-swizzled shared memory, K and V in blocks of BC keys
double-buffered with cp.async. S = Q·Kᵀ reads both operands from shared memory; P stays in registers as the A operand
of P·V, since a warp's accumulator tiles 2kk and 2kk+1 are its A fragment of keys 16kk .. 16kk + 15. Softmax runs in F32.
"""
import math
import sys

BH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
TY = sys.argv[3] if len(sys.argv) > 3 else "f16"
BC = int(sys.argv[4]) if len(sys.argv) > 4 else 128
TM = {"f16": "math.F16", "bf16": "math.BF16"}[TY]
D = 64
BR = 128
QB = L // BR
G = BH * QB
KB = L // BC
SCALE = 1 / math.sqrt(D) * math.log2(math.e)
NT = BC // 8


def i64(x):
    return f"{x}I64"


def i32(x):
    return f"{x}I32"


def fold(op, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = f"{op} ({acc}, {x})"
    return acc


out = []
p = out.append
p(f"""// {TY.upper()} attention on warpgroup MMAs with F32 softmax, {BH} heads of {L} × {D}: 128 query rows per work-group, 64 per
// warpgroup, K and V in blocks of {BC} keys through double-buffered swizzled shared memory; S and P stay in registers.
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
let Frag = «8, 4; F32»;

lam ix (n: Nat) (x: I64): Idx n = core.conv.u n x;
lam shr (x: I64, s: I64): I64 = core.shr.l (x, s);
lam band (x: I64, s: I64): I64 = core.bit2.and 0 (x, s);
lam bxor (x: I64, s: I64): I64 = core.bit2.xor 0 (x, s);
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
lam desc (addr: I32, offset: I64): I64 = gpu.wgmma_desc (core.wrap.add 0 (addr, core.conv.u i32 offset), 16I64, 1024I64, 1I64);

con kernel (m1: gpu.GlobalM, m3: gpu.SharedM, m4: gpu.ConstM, m5: gpu.LocalM,
            group_id: Idx {G}, item_id: Idx 256, (), (q k v o: gpu.GlobalPtr T),
            return: Cn [gpu.GlobalM, gpu.SharedM, gpu.ConstM, gpu.LocalM]) =
    let m       = gpu.generic_mem m1;
    let (m, sq) = gpu.slot_shared «{BR}, 8, 8; H» m;
    let (m, sk) = gpu.slot_shared «2, {BC}, 8, 8; H» m;
    let (m, sv) = gpu.slot_shared «2, {BC}, 8, 8; H» m;
    let addr_q  = gpu.shared_addr sq;
    let addr_k  = gpu.shared_addr sk;
    let addr_v  = gpu.shared_addr sv;
    let qv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 8}, 8; H») q;
    let kv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 8}, 8; H») k;
    let vv      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 8}, 8; H») v;
    let ov      = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 2}, 2; H») o;
    let tid     = core.conv.u 0 item_id;
    let gid     = core.conv.u 0 group_id;
    let h       = shr (gid, {i64(QB.bit_length() - 1)});
    let qrow    = band (gid, {i64(QB - 1)}) * {i64(BR)};
    let wg      = shr (tid, 7I64);
    let lane    = band (tid, 31I64);
    let row0    = qrow + wg * 64I64 + band (shr (tid, 5I64), 3I64) * 16I64;
    let g       = shr (lane, 2I64);
    let t       = band (lane, 3I64);""")
p("""    lam fill (m: mem.M 0, buf: Idx 2, kb: I64): mem.M 0 =""")
for j in range(BC * D // 8 // 256):
    p(f"""        let c{j}      = tid + {i64(256 * j)};
        let r{j}      = shr (c{j}, 3I64);
        let x{j}      = bxor (band (c{j}, 7I64), band (r{j}, 7I64));
        let (m, a{j}) = mem.load (m, mem.lea (mem.lea (mem.lea (kv, ix {BH} h), ix {L} (kb * {i64(BC)} + r{j})), ix {D // 8} (band (c{j}, 7I64))));
        let m        = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sk, buf), ix {BC} r{j}), ix 8 x{j}), a{j});
        let (m, b{j}) = mem.load (m, mem.lea (mem.lea (mem.lea (vv, ix {BH} h), ix {L} (kb * {i64(BC)} + r{j})), ix {D // 8} (band (c{j}, 7I64))));
        let m        = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sv, buf), ix {BC} r{j}), ix 8 x{j}), b{j});""")
p("        gpu.async_commit m;")
for j in range(BR * D // 8 // 256):
    p(f"""    let cq{j}      = tid + {i64(256 * j)};
    let rq{j}      = shr (cq{j}, 3I64);
    let (m, q{j})  = mem.load (m, mem.lea (mem.lea (mem.lea (qv, ix {BH} h), ix {L} (qrow + rq{j})), ix {D // 8} (band (cq{j}, 7I64))));
    let m         = gpu.store_async (m, mem.lea (mem.lea (sq, ix {BR} rq{j}), ix 8 (bxor (band (cq{j}, 7I64), band (rq{j}, 7I64)))), q{j});""")
p(f"""    let base_q = core.wrap.add 0 (addr_q, core.conv.u i32 (wg * 8192I64));
    lam step (m: mem.M 0, buf: Idx 2, o: Frag, mx0 mx1 l0 l1: F32): [mem.M 0, Frag, F32, F32, F32, F32] =
        let base_k = core.wrap.add 0 (addr_k, core.conv.u i32 (core.conv.u 0 buf * {i64(BC * 128)}));
        let base_v = core.wrap.add 0 (addr_v, core.conv.u i32 (core.conv.u 0 buf * {i64(BC * 128)}));
        let m      = gpu.wgmma_fence m;
        let s      = ‹{NT}; ‹4; 0.0:F32››;""")
for ks in range(D // 16):
    p(f"""        let (m, s) = gpu.wgmma_{TY} ({NT}, m, desc (base_q, {i64(32 * ks)}), desc (base_k, {i64(32 * ks)}), ff, s);""")
p(f"""        let m      = gpu.wgmma_commit m;
        let (m, s) = gpu.wgmma_wait (m, 0, s);""")
p(f"""        let r0  = quad_max ({fold("fmax", [f"s#{n}₍{NT}₎#{e}₄" for n in range(NT) for e in (0, 1)])});
        let r1  = quad_max ({fold("fmax", [f"s#{n}₍{NT}₎#{e}₄" for n in range(NT) for e in (2, 3)])});
        let n0  = fmax (mx0, math.arith.mul fast (r0, {SCALE:.9f}:F32));
        let n1  = fmax (mx1, math.arith.mul fast (r1, {SCALE:.9f}:F32));
        let al0 = math.exp.exp2 fast (math.arith.sub fast (mx0, n0));
        let al1 = math.exp.exp2 fast (math.arith.sub fast (mx1, n1));
        lam e (x: F32, mx: F32): F32 = math.exp.exp2 fast (math.arith.sub fast (math.arith.mul fast (x, {SCALE:.9f}:F32), mx));""")
for n in range(NT):
    p(f"""        let p{n} = (e (s#{n}₍{NT}₎#0₄, n0), e (s#{n}₍{NT}₎#1₄, n0), e (s#{n}₍{NT}₎#2₄, n1), e (s#{n}₍{NT}₎#3₄, n1));""")
add = "math.arith.add fast"
p(f"""        let l0 = {add} (math.arith.mul fast (l0, al0), {fold(add, [f"p{n}#{e}₄" for n in range(NT) for e in (0, 1)])});
        let l1 = {add} (math.arith.mul fast (l1, al1), {fold(add, [f"p{n}#{e}₄" for n in range(NT) for e in (2, 3)])});
        let o  = ‹i: 8; (math.arith.mul fast (o#i#0₄, al0), math.arith.mul fast (o#i#1₄, al0),
                         math.arith.mul fast (o#i#2₄, al1), math.arith.mul fast (o#i#3₄, al1))›;
        let m  = gpu.wgmma_fence m;""")
for kk in range(BC // 16):
    regs = ", ".join(f"(narrow (p{2 * kk + r // 2}#{2 * (r % 2)}₄), narrow (p{2 * kk + r // 2}#{2 * (r % 2) + 1}₄))" for r in range(4))
    p(f"""        let (m, o) = gpu.wgmma_rs_{TY} (8, m, ({regs}), desc (base_v, {i64(2048 * kk)}), tt, o);""")
p(f"""        let m      = gpu.wgmma_commit m;
        let (m, o) = gpu.wgmma_wait (m, 0, o);
        (m, o, n0, n1, l0, l1);
    let m = fill (m, 0₂, 0I64);
    affine.For block block_exit (0I64, {i64(KB)}, 1I64, (m, ‹8; ‹4; 0.0:F32››, math.neg_inf math.f32, math.neg_inf math.f32, 0.0:F32, 0.0:F32))
    where
        con block (kb: I64, state: [mem.M 0, Frag, F32, F32, F32, F32], continue: Cn [[mem.M 0, Frag, F32, F32, F32, F32]]) =
            let (m, o, mx0, mx1, l0, l1) = state;
            let m    = gpu.fence_proxy_async (gpu.async_wait_group 0 m);
            let m    = sync (m, m3);
            let next = core.select (core.icmp.ul (kb + 1I64, {i64(KB)}), kb + 1I64, {i64(KB - 1)});
            let m    = fill (m, ix 2 (band (kb + 1I64, 1I64)), next);
            continue (step (m, ix 2 (band (kb, 1I64)), o, mx0, mx1, l0, l1));
        con block_exit (state: [mem.M 0, Frag, F32, F32, F32, F32]) =
            let (m, o, _, _, l0, l1) = state;
            let m  = gpu.async_wait m;
            let inv0 = math.arith.div fast (1.0:F32, quad_sum l0);
            let inv1 = math.arith.div fast (1.0:F32, quad_sum l1);""")
for n in range(D // 8):
    for hh, inv in enumerate(["inv0", "inv1"]):
        p(f"""            let m = mem.store (m, mem.lea (mem.lea (mem.lea (ov, ix {BH} h), ix {L} (row0 + g + {i64(8 * hh)})), ix {D // 2} ({i64(4 * n)} + t)), (narrow (math.arith.mul fast (o#{n}₈#{2 * hh}₄, {inv})), narrow (math.arith.mul fast (o#{n}₈#{2 * hh + 1}₄, {inv}))));""")
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
    ret (m0, m1, m4) = gpu.launch ({G}, 256, gpu.default_stream, ff, ()) kernel (dq, dk, dv, do) $ (m0, m1, m4);
    let (m0, m1)     = gpu.copy_to_host.block (m0, m1, do, o);
    let m1           = mem.free (m1, dq);
    let m1           = mem.free (m1, dk);
    let m1           = mem.free (m1, dv);
    let m1           = mem.free (m1, do);
    let m0           = gpu.deinit (m0, m1, m4);
    return (m0, 0I32);""")
src = "\n".join(out)
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
src = src.replace(f"₍{NT}₎", str(NT).translate(sub))
print(src)
