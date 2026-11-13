"""flash_fa3.py BH L [f16|bf16] [stages] [reg] [ov] [pp]: writes a hand-written warp-specialized 16-bit attention kernel in
Mim for BH heads of L × 64.

The structure is FlashAttention-3's: warpgroup 0 produces, one of its work-items loading Q once and K and V in blocks of
128 keys with TMA into a ring of 128-byte-swizzled shared buffers; warpgroups 1 and 2 consume 64 query rows each with
warpgroup MMAs as in flash_wg.py. Each ring slot has mbarriers for "K full", "V full" and "empty"; nothing else synchronizes.
`reg` moves registers from the producer to the consumers (setmaxnreg); `ov` issues S_j and P_{j-1}·V_{j-1} together and
runs the softmax of S_j while the latter is in flight; `pp` alternates the two consumers' MMAs with named barriers.
"""
import math
import sys

BH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
TY = sys.argv[3] if len(sys.argv) > 3 else "f16"
S = int(sys.argv[4]) if len(sys.argv) > 4 else 2
FLAGS = set(sys.argv[5:])
REG, OV, PP = "reg" in FLAGS, "ov" in FLAGS, "pp" in FLAGS
TM = {"f16": "math.F16", "bf16": "math.BF16"}[TY]
D = 64
BR = 128
BC = 128
QB = L // BR
G = BH * QB
KB = L // BC
LS = S.bit_length() - 1
SCALE = 1 / math.sqrt(D) * math.log2(math.e)
NT = BC // 8
assert S & (S - 1) == 0, "the ring index masks, so the stage count must be a power of two"
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
NTS = str(NT).translate(sub)


def i64(x):
    return f"{x}I64"


def i32(x):
    return f"{x}I32"


def fold(op, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = f"{op} ({acc}, {x})"
    return acc


def indent(lines, n):
    return "\n".join(" " * n + line for line in lines)


add = "math.arith.add fast"
QK = [f"let (m, sc) = gpu.wgmma_{TY} ({NT}, m, desc (base_q, {i64(32 * ks)}), desc (base_k kb, {i64(32 * ks)}), ff, sc);" for ks in range(D // 16)]
PV = [f"let (m, o) = gpu.wgmma_rs_{TY} (8, m, pa#{kk}₈, desc (base_v kb, {i64(2048 * kk)}), tt, o);" for kk in range(BC // 16)]
SOFT = [
    f"let r0  = quad_max ({fold('fmax', [f'sc#{n}{NTS}#{e}₄' for n in range(NT) for e in (0, 1)])});",
    f"let r1  = quad_max ({fold('fmax', [f'sc#{n}{NTS}#{e}₄' for n in range(NT) for e in (2, 3)])});",
    f"let n0  = fmax (mx0, math.arith.mul fast (r0, {SCALE:.9f}:F32));",
    f"let n1  = fmax (mx1, math.arith.mul fast (r1, {SCALE:.9f}:F32));",
    "let al0 = math.exp.exp2 fast (math.arith.sub fast (mx0, n0));",
    "let al1 = math.exp.exp2 fast (math.arith.sub fast (mx1, n1));",
    f"lam e (x: F32, mx: F32): F32 = math.exp.exp2 fast (math.arith.sub fast (math.arith.mul fast (x, {SCALE:.9f}:F32), mx));",
]
for n in range(NT):
    SOFT.append(f"let p{n} = (e (sc#{n}{NTS}#0₄, n0), e (sc#{n}{NTS}#1₄, n0), e (sc#{n}{NTS}#2₄, n1), e (sc#{n}{NTS}#3₄, n1));")
SOFT.append(f"let l0 = {add} (math.arith.mul fast (l0, al0), {fold(add, [f'p{n}#{e}₄' for n in range(NT) for e in (0, 1)])});")
SOFT.append(f"let l1 = {add} (math.arith.mul fast (l1, al1), {fold(add, [f'p{n}#{e}₄' for n in range(NT) for e in (2, 3)])});")
frags = ", ".join(
    "(" + ", ".join(f"(narrow (p{2 * kk + r // 2}#{2 * (r % 2)}₄), narrow (p{2 * kk + r // 2}#{2 * (r % 2) + 1}₄))" for r in range(4)) + ")"
    for kk in range(BC // 16))
SOFT.append(f"let pa = ({frags});")
STORE = []
for n in range(D // 8):
    for hh, inv in enumerate(["inv0", "inv1"]):
        STORE.append(f"let m = mem.store (m, mem.lea (mem.lea (mem.lea (ov, ix {BH} h), ix {L} (row0 + g + {i64(8 * hh)})), ix {D // 2} ({i64(4 * n)} + t)), (narrow (math.arith.mul fast (o#{n}₈#{2 * hh}₄, {inv})), narrow (math.arith.mul fast (o#{n}₈#{2 * hh + 1}₄, {inv}))));")

inits = "gpu.mbar_init (m, full_q, 1I32)"
for s in range(S):
    inits = f"gpu.mbar_init (gpu.mbar_init (gpu.mbar_init ({inits}, full_k {i64(s)}, 1I32), full_v {i64(s)}, 1I32), empty {i64(s)}, 256I32)"

if OV:
    # Iteration kb issues S_kb and P_{kb-1}·V_{kb-1}, then runs softmax(S_kb) while the latter is in flight.
    LOOP = f"""let (m, sc)                   = qk (turn m, 0I64);
let (m, sc)                   = gpu.wgmma_wait (m, 0, sc);
let m                         = pass m;
let (pa, mx0, mx1, l0, l1, _, _) = soft (sc, math.neg_inf math.f32, math.neg_inf math.f32, 0.0:F32, 0.0:F32);
affine.For block block_exit (1I64, {i64(KB)}, 1I64, (m, ‹8; ‹4; 0.0:F32››, pa, mx0, mx1, l0, l1))
where
    con block (kb: I64, state: State, continue: Cn State) =
        let (m, o, pa, mx0, mx1, l0, l1) = state;
        let (m, sc) = qk (turn m, kb);
        let (m, o)  = pv (m, kb - 1I64, pa, o);
        let m       = pass m;
        let (m, sc) = gpu.wgmma_wait (m, 1, sc);
        let (np, n0, n1, l0, l1, al0, al1) = soft (sc, mx0, mx1, l0, l1);
        let (m, ow) = gpu.wgmma_wait (m, 0, (o, pa));
        let (o, _)  = ow;
        let m       = gpu.mbar_arrive (m, empty (slot (kb - 1I64)));
        continue (m, rescale (o, al0, al1), np, n0, n1, l0, l1);"""
    TAIL = f"""let (m, o)  = pv (m, {i64(KB - 1)}, pa, o);
let (m, ow) = gpu.wgmma_wait (m, 0, (o, pa));
let (o, _)  = ow;
let m       = gpu.mbar_arrive (m, empty (slot {i64(KB - 1)}));"""
else:
    LOOP = f"""affine.For block block_exit (0I64, {i64(KB)}, 1I64, (m, ‹8; ‹4; 0.0:F32››, ‹8; ‹4; ‹2; narrow 0.0:F32›››, math.neg_inf math.f32, math.neg_inf math.f32, 0.0:F32, 0.0:F32))
where
    con block (kb: I64, state: State, continue: Cn State) =
        let (m, o, _, mx0, mx1, l0, l1) = state;
        let (m, sc) = qk (turn m, kb);
        let (m, sc) = gpu.wgmma_wait (m, 0, sc);
        let (pa, n0, n1, l0, l1, al0, al1) = soft (sc, mx0, mx1, l0, l1);
        let (m, o)  = pv (m, kb, pa, rescale (o, al0, al1));
        let m       = pass m;
        let (m, ow) = gpu.wgmma_wait (m, 0, (o, pa));
        let (o, _)  = ow;
        let m       = gpu.mbar_arrive (m, empty (slot kb));
        continue (m, o, pa, n0, n1, l0, l1);"""
    TAIL = ""

src = f"""// Warp-specialized {TY.upper()} attention with F32 softmax, {BH} heads of {L} × {D}: a TMA producer warpgroup and two consumer
// warpgroups of 64 query rows, K and V in blocks of {BC} keys through a {S}-slot ring synchronized by mbarriers ({" ".join(sorted(FLAGS)) or "plain"}).
plugin core;
plugin gpu;
plugin mem;
plugin math;
plugin affine;

use core.ops.u.w;

let H     = {TM};
let F32   = math.F32;
let fast  = math.mode.fast;
let T     = «{BH}, {L}, {D}; H»;
let TMap  = «16; I64»;
let Frag  = «8, 4; F32»;
let Sc    = «{NT}, 4; F32»;
let Pa    = «{BC // 16}, 4, 2; H»;
let State = [mem.M 0, Frag, Pa, F32, F32, F32, F32];

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
lam desc (addr: I32, offset: I64): I64 = gpu.wgmma_desc (core.wrap.add 0 (addr, core.conv.u i32 offset), 16I64, 1024I64, 1I64);
lam w32 (x: I64): I32 = core.conv.u i32 x;

con kernel (m1: gpu.GlobalM, m3: gpu.SharedM, m4: gpu.ConstM, m5: gpu.LocalM,
            group_id: Idx {G}, item_id: Idx 384, (), (mq mk mv: gpu.GlobalPtr TMap, o: gpu.GlobalPtr T),
            return: Cn [gpu.GlobalM, gpu.SharedM, gpu.ConstM, gpu.LocalM]) =
    let m         = gpu.generic_mem m1;
    let (m, sq)   = gpu.slot_shared «{BR}, 8, 8; H» m;
    let (m, sk)   = gpu.slot_shared «{S}, {BC}, 8, 8; H» m;
    let (m, sv)   = gpu.slot_shared «{S}, {BC}, 8, 8; H» m;
    let (m, bars) = gpu.slot_shared «{3 * S + 1}; I64» m;
    let full_q    = mem.lea (bars, 0{str(3 * S + 1).translate(sub)});
    lam full_k (s: I64): mem.Ptr0 I64 = mem.lea (bars, ix {3 * S + 1} (1I64 + s));
    lam full_v (s: I64): mem.Ptr0 I64 = mem.lea (bars, ix {3 * S + 1} ({i64(1 + S)} + s));
    lam empty (s: I64): mem.Ptr0 I64 = mem.lea (bars, ix {3 * S + 1} ({i64(1 + 2 * S)} + s));
    let addr_q    = gpu.shared_addr sq;
    let addr_k    = gpu.shared_addr sk;
    let addr_v    = gpu.shared_addr sv;
    let ov        = core.bitcast (mem.Ptr0 «{BH}, {L}, {D // 2}, 2; H») o;
    let tid       = core.conv.u 0 item_id;
    let gid       = core.conv.u 0 group_id;
    let h         = shr (gid, {i64(QB.bit_length() - 1)});
    let qrow      = band (gid, {i64(QB - 1)}) * {i64(BR)};
    let wg        = shr (tid, 7I64);
    (skip, init)#(tid == 0I64) m
    where
        con init (m: mem.M 0) = ready ({inits});
        con skip (m: mem.M 0) = ready m;
        con ready (m: mem.M 0) =
            let (m1, m3) = gpu.sync_work_items (gpu.global_mem m, m3);
            (consume, produce)#(wg == 0I64) (gpu.generic_mem m1)
            where
                con produce (m: mem.M 0) = (done, lead)#(tid == 0I64) {"(gpu.setmaxnreg (m, ff, 40))" if REG else "m"}
                where
                    con lead (m: mem.M 0) =
                        let m = gpu.mbar_expect_tx (m, full_q, {i32(BR * D * 2)});
                        let m = gpu.tma_load (m, sq, mq, 0I32, w32 (h * {i64(L)} + qrow), full_q);
                        affine.For load load_exit (0I64, {i64(KB)}, 1I64, m)
                        where
                            con load (kb: I64, m: mem.M 0, continue: Cn mem.M 0) =
                                let s   = band (kb, {i64(S - 1)});
                                let m   = gpu.mbar_wait (m, empty s, w32 (core.bit2.xor 0 (band (shr (kb, {i64(LS)}), 1I64), 1I64)));
                                let row = w32 (h * {i64(L)} + kb * {i64(BC)});
                                let m   = gpu.mbar_expect_tx (m, full_k s, {i32(BC * D * 2)});
                                let m   = gpu.tma_load (m, mem.lea (sk, ix {S} s), mk, 0I32, row, full_k s);
                                let m   = gpu.mbar_expect_tx (m, full_v s, {i32(BC * D * 2)});
                                let m   = gpu.tma_load (m, mem.lea (sv, ix {S} s), mv, 0I32, row, full_v s);
                                continue m;
                            con load_exit (m: mem.M 0) = done m;
                        end;
                    con done (m: mem.M 0) = return (gpu.global_mem m, m3, m4, m5);
                end;
                con consume (m: mem.M 0) =
                    let m      = {"gpu.setmaxnreg (m, tt, 232)" if REG else "m"};
                    let c      = wg - 1I64;
                    let lane   = band (tid, 31I64);
                    let row0   = qrow + c * 64I64 + band (shr (tid, 5I64), 3I64) * 16I64;
                    let g      = shr (lane, 2I64);
                    let t      = band (lane, 3I64);
                    let base_q = core.wrap.add 0 (addr_q, w32 (c * 8192I64));
                    lam slot (kb: I64): I64 = band (kb, {i64(S - 1)});
                    lam parity (kb: I64): I32 = w32 (band (shr (kb, {i64(LS)}), 1I64));
                    lam base_k (kb: I64): I32 = core.wrap.add 0 (addr_k, w32 (slot kb * {i64(BC * 128)}));
                    lam base_v (kb: I64): I32 = core.wrap.add 0 (addr_v, w32 (slot kb * {i64(BC * 128)}));
                    lam turn (m: mem.M 0): mem.M 0 = {"gpu.sync_named (m, w32 (1I64 + c), 256I32)" if PP else "m"};
                    lam pass (m: mem.M 0): mem.M 0 = {"gpu.arrive_named (m, w32 (2I64 - c), 256I32)" if PP else "m"};
                    lam qk (m: mem.M 0, kb: I64): [mem.M 0, Sc] =
                        let m  = gpu.mbar_wait (m, full_k (slot kb), parity kb);
                        let m  = gpu.wgmma_fence m;
                        let sc = ‹{NT}; ‹4; 0.0:F32››;
{indent(QK, 24)}
                        (gpu.wgmma_commit m, sc);
                    lam pv (m: mem.M 0, kb: I64, pa: Pa, o: Frag): [mem.M 0, Frag] =
                        let m = gpu.mbar_wait (m, full_v (slot kb), parity kb);
                        let m = gpu.wgmma_fence m;
{indent(PV, 24)}
                        (gpu.wgmma_commit m, o);
                    lam soft (sc: Sc, mx0 mx1 l0 l1: F32): [Pa, F32, F32, F32, F32, F32, F32] =
{indent(SOFT, 24)}
                        (pa, n0, n1, l0, l1, al0, al1);
                    lam rescale (o: Frag, al0 al1: F32): Frag =
                        ‹i: 8; (math.arith.mul fast (o#i#0₄, al0), math.arith.mul fast (o#i#1₄, al0),
                                math.arith.mul fast (o#i#2₄, al1), math.arith.mul fast (o#i#3₄, al1))›;
                    (start, first)#(c == 1I64) m
                    where
                    con first (m: mem.M 0) = start {"(gpu.arrive_named (m, 1I32, 256I32))" if PP else "m"};
                    con start (m: mem.M 0) =
                    let m = gpu.mbar_wait (m, full_q, 0I32);
{indent(LOOP.splitlines(), 20)}
        con block_exit (state: State) =
            let (m, o, pa, _, _, l0, l1) = state;
{indent(TAIL.splitlines(), 12)}
            let inv0 = math.arith.div fast (1.0:F32, quad_sum l0);
            let inv1 = math.arith.div fast (1.0:F32, quad_sum l1);
{indent(STORE, 12)}
            return (gpu.global_mem m, m3, m4, m5);
    end;
                    end;
            end;
    end;

extern fun flash (m0: mem.M 0, q k v o: mem.Ptr0 T): [mem.M 0, I32] =
    let (m0, m1, m4) = gpu.init m0;
    let (m0, m1, dq) = gpu.alloc_copy.block (m0, m1, q);
    let (m0, m1, dk) = gpu.alloc_copy.block (m0, m1, k);
    let (m0, m1, dv) = gpu.alloc_copy.block (m0, m1, v);
    let (m1, do)     = mem.alloc (T, 1) m1;
    let (m0, hq)     = mem.alloc (TMap, 0) m0;
    let (m0, hk)     = mem.alloc (TMap, 0) m0;
    let (m0, hv)     = mem.alloc (TMap, 0) m0;
    let m0           = gpu.tensor_map H (m0, hq, dq, {i64(BH * L)}, {i64(D)}, {i32(BR)}, {i32(D)});
    let m0           = gpu.tensor_map H (m0, hk, dk, {i64(BH * L)}, {i64(D)}, {i32(BC)}, {i32(D)});
    let m0           = gpu.tensor_map H (m0, hv, dv, {i64(BH * L)}, {i64(D)}, {i32(BC)}, {i32(D)});
    let (m0, m1, mq) = gpu.alloc_copy.block (m0, m1, hq);
    let (m0, m1, mk) = gpu.alloc_copy.block (m0, m1, hk);
    let (m0, m1, mv) = gpu.alloc_copy.block (m0, m1, hv);
    ret (m0, m1, m4) = gpu.launch ({G}, 384, gpu.default_stream, ff, ()) kernel (mq, mk, mv, do) $ (m0, m1, m4);
    let (m0, m1)     = gpu.copy_to_host.block (m0, m1, do, o);
    let m1           = mem.free (m1, dq);
    let m1           = mem.free (m1, dk);
    let m1           = mem.free (m1, dv);
    let m1           = mem.free (m1, do);
    let m1           = mem.free (m1, mq);
    let m1           = mem.free (m1, mk);
    let m1           = mem.free (m1, mv);
    let m0           = mem.free (m0, hq);
    let m0           = mem.free (m0, hk);
    let m0           = mem.free (m0, hv);
    let m0           = gpu.deinit (m0, m1, m4);
    return (m0, 0I32);"""
print(src)
