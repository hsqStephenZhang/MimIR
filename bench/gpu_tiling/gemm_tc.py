"""gemm_tc.py [n]: writes a hand-written TF32 tensor-core SGEMM in Mim: 128x128 blocks, BK=16, 8 warps of 64x32, cp.async double buffering."""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
GN = N // 128
G = GN * GN
KB = N // 16
LG = GN.bit_length() - 1

def i64(x): return f"{x}I64"

out = []
p = out.append
p(f"""// TF32 tensor-core SGEMM, {N}^3: 128x128 work-group tiles, k in blocks of 16 double-buffered with cp.async,
// 8 warps of 64x32, A fragments via ldmatrix.x4, B fragments via scalar shared loads.
plugin core;
plugin gpu;
plugin mem;
plugin math;
plugin affine;

use core.ops.u.w;

let F32 = math.F32;
let Mat = «{N}, {N}; F32»;
let Acc = «16; «4; F32»»;

lam ix (n: Nat) (x: I64): Idx n = core.conv.u n x;
lam shr (x: I64, s: I64): I64 = core.shr.l (x, s);
lam band (x: I64, s: I64): I64 = core.bit2.and 0 (x, s);
lam sync (m: mem.M 0, m3: gpu.SharedM): mem.M 0 =
    let (m1, _) = gpu.sync_work_items (gpu.global_mem m, m3);
    gpu.generic_mem m1;

con kernel (m1: gpu.GlobalM, m3: gpu.SharedM, m4: gpu.ConstM, m5: gpu.LocalM,
            group_id: Idx {G}, item_id: Idx 256, (), (a b c: gpu.GlobalPtr Mat),
            return: Cn [gpu.GlobalM, gpu.SharedM, gpu.ConstM, gpu.LocalM]) =
    let m      = gpu.generic_mem m1;
    let (m, sa) = gpu.slot_shared «2, 128, 5, 4; F32» m;
    let (m, sb) = gpu.slot_shared «2, 16, 34, 4; F32» m;
    let sbs    = core.bitcast (mem.Ptr0 «2, 16, 136; F32») sb;
    let av     = core.bitcast (mem.Ptr0 «{N}, {N // 4}, 4; F32») a;
    let bv     = core.bitcast (mem.Ptr0 «{N}, {N // 4}, 4; F32») b;
    let cv     = core.bitcast (mem.Ptr0 Mat) c;
    let tid    = core.conv.u 0 item_id;
    let gid    = core.conv.u 0 group_id;
    let bm     = shr (gid, {i64(LG)});
    let bn     = band (gid, {i64(GN - 1)});
    let lane   = band (tid, 31I64);
    let warp   = shr (tid, 5I64);
    let wm     = shr (warp, 2I64);
    let wn     = band (warp, 3I64);
    let g      = shr (lane, 2I64);
    let t      = band (lane, 3I64);
    lam fill (m: mem.M 0, buf: Idx 2, kb: I64): mem.M 0 =""")
for j in range(2):
    p(f"""        let ca{j}       = tid + {i64(256 * j)};
        let pa{j}       = mem.lea (mem.lea (av, ix {N} (bm * 128I64 + shr (ca{j}, 2I64))), ix {N // 4} (kb * 4I64 + band (ca{j}, 3I64)));
        let (m, va{j})  = mem.load (m, pa{j});
        let m          = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sa, buf), ix 128 (shr (ca{j}, 2I64))), ix 5 (band (ca{j}, 3I64))), va{j});
        let pb{j}       = mem.lea (mem.lea (bv, ix {N} (kb * 16I64 + shr (ca{j}, 5I64))), ix {N // 4} (bn * 32I64 + band (ca{j}, 31I64)));
        let (m, vb{j})  = mem.load (m, pb{j});
        let m          = gpu.store_async (m, mem.lea (mem.lea (mem.lea (sb, buf), ix 16 (shr (ca{j}, 5I64))), ix 34 (band (ca{j}, 31I64))), vb{j});""")
p("        m;")
p("""    lam step (m: mem.M 0, buf: Idx 2, acc: Acc): [mem.M 0, Acc] =""")
for ks in range(2):
    for mi in range(4):
        p(f"""        let (m, a{ks}{mi}) = gpu.ldmatrix (4, m, mem.lea (mem.lea (mem.lea (sa, buf), ix 128 (wm * 64I64 + {i64(mi * 16)} + band (lane, 15I64))), ix 5 ({i64(ks * 2)} + shr (lane, 4I64))));""")
    for ni in range(4):
        for h in range(2):
            p(f"""        let (m, b{ks}{ni}{h}) = mem.load (m, mem.lea (mem.lea (mem.lea (sbs, buf), ix 16 ({i64(ks * 8 + 4 * h)} + t)), ix 136 (wn * 32I64 + {i64(ni * 8)} + g)));""")
    p(f"""        let af{ks} = (a{ks}0, a{ks}1, a{ks}2, a{ks}3);
        let bf{ks} = ((b{ks}00, b{ks}01), (b{ks}10, b{ks}11), (b{ks}20, b{ks}21), (b{ks}30, b{ks}31));
        let acc = ‹i: 16; gpu.mma_tf32 (af{ks}#(core.conv.u 4 (core.shr.l (i, 2₁₆))), bf{ks}#(core.conv.u 4 (core.bit2.and 0 (i, 3₁₆))), acc#i)›;""")
p("        (m, acc);")
p(f"""    let m = fill (m, 0₂, 0I64);
    affine.For block block_exit (0I64, {i64(KB)}, 1I64, (m, ‹16; ‹4; 0.0:F32››))
    where
        con block (kb: I64, state: [mem.M 0, Acc], continue: Cn [[mem.M 0, Acc]]) =
            let (m, acc) = state;
            let m        = sync (gpu.async_wait m, m3);
            let buf      = ix 2 (band (kb, 1I64));
            let next     = core.select (core.icmp.ul (kb + 1I64, {i64(KB)}), kb + 1I64, {i64(KB - 1)});
            let m        = fill (m, ix 2 (band (kb + 1I64, 1I64)), next);
            let (m, acc) = step (m, buf, acc);
            continue (m, acc);
        con block_exit (state: [mem.M 0, Acc]) =
            let (m, acc) = state;
            let m        = gpu.async_wait m;""")
for mi in range(4):
    for ni in range(4):
        i = mi * 4 + ni
        p(f"""            let r{i} = bm * 128I64 + wm * 64I64 + {i64(mi * 16)} + g;
            let c{i} = bn * 128I64 + wn * 32I64 + {i64(ni * 8)} + t * 2I64;""")
        for e, (dr, dc) in enumerate([(0, 0), (0, 1), (8, 0), (8, 1)]):
            p(f"""            let m = mem.store (m, mem.lea (mem.lea (cv, ix {N} (r{i} + {i64(dr)})), ix {N} (c{i} + {i64(dc)})), acc#{i}₁₆#{e}₄);""")
p("""            return (gpu.global_mem m, m3, m4, m5);
    end;
""")
p(f"""extern fun gemm_tc (m0: mem.M 0, a b c: mem.Ptr0 Mat): [mem.M 0, I32] =
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
