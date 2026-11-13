"""gemm_wg.py [n] [f16|bf16] [stages]: writes a hand-written 16-bit WGMMA GEMM in Mim with F32 accumulation: 128x128 blocks of two
warpgroups, BK=64, A and B through 128-byte-swizzled shared memory filled with cp.async in a multi-stage ring."""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TY = sys.argv[2] if len(sys.argv) > 2 else "f16"
S = int(sys.argv[3]) if len(sys.argv) > 3 else 4
T = {"f16": "math.F16", "bf16": "math.BF16"}[TY]
GN = N // 128
G = GN * GN
KB = N // 64
LG = GN.bit_length() - 1
assert S & (S - 1) == 0, "the ring index masks, so the stage count must be a power of two"

def i64(x): return f"{x}I64"
def i32(x): return f"{x}I32"
def sub(x): return str(x).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

out = []
p = out.append
p(f"""// {TY.upper()} WGMMA GEMM with F32 accumulation, {N}^3: 128x128 work-group tiles of two warpgroups, k in blocks of 64 through
// a {S}-stage cp.async ring of 128-byte-swizzled shared memory; A is K-major, B MN-major.
plugin core;
plugin gpu;
plugin mem;
plugin math;
plugin affine;

use core.ops.u.w;

let T   = {T};
let F32 = math.F32;
let Mat = «{N}, {N}; T»;
let Acc = «16, 4; F32»;

lam ix (n: Nat) (x: I64): Idx n = core.conv.u n x;
lam shr (x: I64, s: I64): I64 = core.shr.l (x, s);
lam band (x: I64, s: I64): I64 = core.bit2.and 0 (x, s);
lam bxor (x: I64, s: I64): I64 = core.bit2.xor 0 (x, s);
lam sync (m: mem.M 0, m3: gpu.SharedM): mem.M 0 =
    let (m1, _) = gpu.sync_work_items (gpu.global_mem m, m3);
    gpu.generic_mem m1;
lam narrow (x: F32): T = math.conv.f2f math.{TY} x;

con kernel (m1: gpu.GlobalM, m3: gpu.SharedM, m4: gpu.ConstM, m5: gpu.LocalM,
            group_id: Idx {G}, item_id: Idx 256, (), (a b c: gpu.GlobalPtr Mat),
            return: Cn [gpu.GlobalM, gpu.SharedM, gpu.ConstM, gpu.LocalM]) =
    let m       = gpu.generic_mem m1;
    let (m, sa) = gpu.slot_shared «{S}, 128, 8, 8; T» m;
    let (m, sb) = gpu.slot_shared «{S}, 2, 64, 8, 8; T» m;
    let av      = core.bitcast (mem.Ptr0 «{N}, {N // 8}, 8; T») a;
    let bv      = core.bitcast (mem.Ptr0 «{N}, {N // 8}, 8; T») b;
    let cv      = core.bitcast (mem.Ptr0 «{N}, {N // 2}, 2; T») c;
    let addr_a  = gpu.shared_addr sa;
    let addr_b  = gpu.shared_addr sb;
    let tid     = core.conv.u 0 item_id;
    let gid     = core.conv.u 0 group_id;
    let bm      = shr (gid, {i64(LG)});
    let bn      = band (gid, {i64(GN - 1)});
    let wg      = shr (tid, 7I64);
    let lane    = band (tid, 31I64);
    let warp    = band (shr (tid, 5I64), 3I64);
    let g       = shr (lane, 2I64);
    let t       = band (lane, 3I64);
    lam fill (m: mem.M 0, buf: Idx {S}, kb: I64): mem.M 0 =""")
for j in range(4):
    p(f"""        let ca{j}      = tid + {i64(256 * j)};
        let ra{j}      = shr (ca{j}, 3I64);
        let (m, va{j}) = mem.load (m, mem.lea (mem.lea (av, ix {N} (bm * 128I64 + ra{j})), ix {N // 8} (kb * 8I64 + band (ca{j}, 7I64))));
        let m         = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sa, buf), ix 128 ra{j}), ix 8 (bxor (band (ca{j}, 7I64), band (ra{j}, 7I64)))), va{j});
        let kr{j}      = shr (ca{j}, 4I64);
        let (m, vb{j}) = mem.load (m, mem.lea (mem.lea (bv, ix {N} (kb * 64I64 + kr{j})), ix {N // 8} (bn * 16I64 + band (ca{j}, 15I64))));
        let m         = gpu.store_async (m, mem.lea (mem.lea (mem.lea (mem.lea (sb, buf), ix 2 (band (shr (ca{j}, 3I64), 1I64))), ix 64 kr{j}), ix 8 (bxor (band (ca{j}, 7I64), band (kr{j}, 7I64)))), vb{j});""")
p("        gpu.async_commit m;")
p(f"""    lam step (m: mem.M 0, buf: Idx {S}, acc: Acc): [mem.M 0, Acc] =
        let base_a = core.wrap.add 0 (addr_a, core.conv.u i32 (core.conv.u 0 buf * 16384I64 + wg * 8192I64));
        let base_b = core.wrap.add 0 (addr_b, core.conv.u i32 (core.conv.u 0 buf * 16384I64));
        let m      = gpu.wgmma_fence m;""")
for ks in range(4):
    p(f"""        let (m, acc) = gpu.wgmma_{TY} (16, m, gpu.wgmma_desc (core.wrap.add 0 (base_a, {i32(32 * ks)}), 16I64, 1024I64, 1I64),
                                     gpu.wgmma_desc (core.wrap.add 0 (base_b, {i32(2048 * ks)}), 8192I64, 1024I64, 1I64), tt, acc);""")
p("""        let m = gpu.wgmma_commit m;
        gpu.wgmma_wait (m, 0, acc);""")
pro = "m"
for s in range(S - 1):
    pro = f"fill ({pro}, {s}{sub(S)}, {i64(min(s, KB - 1))})"
p(f"""    let m = {pro};
    affine.For block block_exit (0I64, {i64(KB)}, 1I64, (m, ‹16; ‹4; 0.0:F32››))
    where
        con block (kb: I64, state: [mem.M 0, Acc], continue: Cn [[mem.M 0, Acc]]) =
            let (m, acc) = state;
            let m        = gpu.fence_proxy_async (gpu.async_wait_group {S - 2} m);
            let m        = sync (m, m3);
            let ahead    = kb + {i64(S - 1)};
            let next     = core.select (core.icmp.ul (ahead, {i64(KB)}), ahead, {i64(KB - 1)});
            let m        = fill (m, ix {S} (band (ahead, {i64(S - 1)})), next);
            let (m, acc) = step (m, ix {S} (band (kb, {i64(S - 1)})), acc);
            continue (m, acc);
        con block_exit (state: [mem.M 0, Acc]) =
            let (m, acc) = state;
            let m        = gpu.async_wait m;
            let row      = bm * 128I64 + wg * 64I64 + warp * 16I64 + g;""")
for j in range(16):
    for h in range(2):
        p(f"""            let m = mem.store (m, mem.lea (mem.lea (cv, ix {N} (row + {i64(8 * h)})), ix {N // 2} (bn * 64I64 + {i64(4 * j)} + t)), (narrow (acc#{j}₁₆#{2 * h}₄), narrow (acc#{j}₁₆#{2 * h + 1}₄)));""")
p("""            return (gpu.global_mem m, m3, m4, m5);
    end;
""")
p(f"""extern fun gemm_wg (m0: mem.M 0, a b c: mem.Ptr0 Mat): [mem.M 0, I32] =
    let (m0, m1, m4) = gpu.init m0;
    let (m1, da)     = mem.alloc (Mat, 1) m1;
    let (m1, db)     = mem.alloc (Mat, 1) m1;
    let (m1, dc)     = mem.alloc (Mat, 1) m1;
    let (m0, m1)     = gpu.copy_to_device.block (m0, m1, a, da);
    let (m0, m1)     = gpu.copy_to_device.block (m0, m1, b, db);
    ret (m0, m1, m4) = gpu.launch ({G}, 256, gpu.default_stream, ff, ()) kernel (da, db, dc) $ (m0, m1, m4);
    let (m0, m1)     = gpu.copy_to_host.block (m0, m1, dc, c);
    let m1           = mem.free (m1, da);
    let m1           = mem.free (m1, db);
    let m1           = mem.free (m1, dc);
    let m0           = gpu.deinit (m0, m1, m4);
    return (m0, 0I32);""")
print("\n".join(out))
