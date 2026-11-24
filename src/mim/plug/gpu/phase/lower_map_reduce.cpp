#include "mim/plug/gpu/phase/lower_map_reduce.h"

#include <algorithm>

#include <fe/log.h>
#include <fe/vector.h>

#include <mim/check.h>
#include <mim/driver.h>
#include <mim/lam.h>
#include <mim/rewrite.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/btensor/btensor.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

namespace {

using fe::Vector;

/// Whether an explicit, user-written `gpu.init` is reachable from `def`.
bool contains_gpu_init(const Def* def, DefSet& seen) {
    if (auto [_, ins] = seen.emplace(def); !ins) return false;
    if (Axm::isa<gpu::init>(def)) return true;
    for (auto d : def->deps())
        if (contains_gpu_init(d, seen)) return true;
    return false;
}

/// Substitutes the var of @p old_lam (mapping it to @p var_repl) and moves every memory in its Nest to `gpu.GlobalM`.
class GlobalMem : public Rewriter {
public:
    GlobalMem(Lam* old_lam, const Def* var_repl)
        : Rewriter(old_lam->world())
        , old_lam_(old_lam) {
        auto& w = world();
        map(old_lam->var(), var_repl);
        map(w.call<mem::M>(0), w.annex<gpu::GlobalM>());
    }

    const Def* rewrite(const Def* old_def) final {
        if (auto mut = old_def->isa_mut(); mut && !old_lam_->nests(mut)) return mut;
        return Rewriter::rewrite(old_def);
    }

private:
    Lam* old_lam_;
};

/// Rebuilds `comb`/`post` as a canonical 2-var `Fn [GlobalM, Tin, extra] → [GlobalM, Tout]` even if scalarize flattened it.
Lam* rebuild_lam_global_mem(Lam* lam, const Def* Tout, Sym name) {
    auto& w        = lam->world();
    auto global_ty = w.annex<gpu::GlobalM>();

    auto nv = lam->num_vars();
    const Def* Tin;
    const Def* extra_ty;
    if (nv == 2) {
        auto [_, t, e] = lam->var(nv, 0)->type()->projs<3>();
        Tin            = t;
        extra_ty       = e;
    } else {
        Tin      = lam->var(nv, 1)->type();
        extra_ty = lam->var(nv, 2)->type();
    }
    auto new_lam = w.mut_con(Defs{w.sigma({global_ty, Tin, extra_ty}), w.cn({global_ty, Tout})})->set(name);
    const Def* repl = new_lam->var();
    if (nv != 2) {
        auto [m, t, e] = new_lam->var(2, 0)->projs<3>();
        repl           = w.tuple({m, t, e, new_lam->var(2, 1)});
    }
    new_lam->set(true, GlobalMem(lam, repl).rewrite(lam->body()));
    return new_lam;
}

/// The row-major coordinates of @p flat - an `I64`, i.e. an `Idx 0` - via `affine.delinearize`.
std::pair<const Def*, DefVec> unflatten_index(World& w, const Def* flat, const Vector<nat_t>& dims, const Def* mem) {
    auto m = dims.size();
    if (m == 0) return {mem, {}};

    auto s_out = w.tuple(DefVec(m, [&](size_t d) { return w.lit_nat(dims[d]); }));
    auto index = w.annex<affine::Index>();
    auto f     = w.mut_lam(index, w.arr(m, index))->set("unflatten");
    f->set(true, w.app(w.app(w.annex<affine::delinearize>(), w.lit_nat(m)), {f->var(), s_out}));

    auto [mem2, coords]
        = btensor::op_affine_map(f, w.lit_nat(m), w.lit_nat_1(), w.lit_nat_0(), s_out, flat, mem)->projs<2>();
    return {mem2, coords->projs(m)};
}

struct InputDesc {
    DefVec rs, ss, ts, maps;
};

InputDesc extract_input_desc(nat_t n, const Def* Rs, const Def* Ss, const Def* Ts, const Def* maps) {
    InputDesc desc{DefVec(n), DefVec(n), DefVec(n), DefVec(n)};
    for (nat_t i = 0; i != n; ++i) {
        desc.rs[i]   = Rs->proj(n, i);
        desc.ss[i]   = Ss->proj(n, i);
        desc.ts[i]   = Ts->proj(n, i);
        desc.maps[i] = maps->proj(n, i);
    }
    return desc;
}

struct Inputs {
    const Def* mem;
    const Def* global;
    DefVec dptrs;
};

Inputs alloc_copy_inputs(World& w, const Def* m0, const Def* m1, Defs ris, Defs sis, Defs tis, const Def* inputs) {
    DefVec dptrs(ris.size());
    for (size_t i = 0; i != ris.size(); ++i) {
        auto alloc_copy    = w.app(w.app(w.annex<gpu::buf_alloc_copy>(), {ris[i], sis[i], tis[i]}),
                                   {m0, m1, inputs->proj(ris.size(), i)});
        auto [m2, g2, ptr] = alloc_copy->projs<3>();
        m0                 = m2;
        m1                 = g2;
        dptrs[i]           = ptr;
    }
    return {m0, m1, dptrs};
}

std::pair<const Def*, const Def*> alloc_output(World& w, const Def* m1, const Def* elem_ty, const Def* So) {
    return w.app(w.app(w.annex<gpu::alloc>(gpu::alloc::block), w.arr(So, elem_ty)), m1)->projs<2>();
}

struct Grid {
    nat_t n_groups, n_items, total;
};

Grid grid_layout(const Vector<nat_t>& out_dims) {
    nat_t total = 1;
    for (auto d : out_dims)
        total *= d;
    nat_t n_items  = std::min<nat_t>(total, 1024);
    nat_t n_groups = (total + n_items - 1) / n_items;
    return {n_groups, n_items, total};
}

/// Per-input state for `build_kernel`: `InputDesc`'s shapes/access-functions plus `alloc_copy_inputs`'s pointers.
struct Mapped {
    DefVec rs, ss, dptrs, maps;
    nat_t n() const { return dptrs.size(); }
};

/// Builds the kernel: one thread per output point, reducing sequentially over the `rr` reduction dims.
Lam* build_kernel(World& w,
                  const Def* Ro,
                  nat_t rr,
                  const Vector<nat_t>& out_dims,
                  const Def* Sr,
                  const Def* So,
                  const Mapped& ins,
                  const Def* To,
                  const Def* map_out,
                  const Def* init,
                  Lam* global_comb,
                  const Mapped& post_ins,
                  Lam* global_post,
                  const Def* Tp,
                  const Def* out_dptr,
                  const Grid& grid) {
    auto nis        = ins.n();
    auto nps        = post_ins.n();
    auto ro         = out_dims.size();
    auto nloops_nat = ro + rr;
    auto n          = w.lit_nat(nloops_nat);

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("mapReduceKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);

    auto group_i64 = grid.n_groups == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), group_id);
    auto item_i64  = grid.n_items == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), item_id);
    auto flat
        = w.call(core::wrap::add, core::Mode::none,
                 Defs{w.call(core::wrap::mul, core::Mode::none, Defs{group_i64, w.lit_i64(grid.n_items)}), item_i64});

    auto in_range     = w.call(core::icmp::ul, Defs{flat, w.lit_i64(grid.total)});
    auto early_return = w.mut_con(w.sigma(Defs{}))->set("outOfRange");
    early_return->app(true, k_ret, Defs{k_global, k_shared, k_const, k_local});
    auto body = w.mut_con(w.sigma(Defs{}))->set("inRange");
    kernel->branch(true, in_range, body, early_return);

    auto write_back           = w.mut_con(Defs{global_ty, To})->set("writeBack");
    auto [wb_mem, acc_final]  = write_back->vars<2>();
    auto [wb_mem2, wb_coords] = unflatten_index(w, flat, out_dims, wb_mem);
    DefVec wb_idx             = wb_coords;
    for (size_t j = 0; j != rr; ++j)
        wb_idx.push_back(w.call(core::conv::u, Sr->proj(nloops_nat, ro + j), w.lit_i64(0)));
    auto [wc_mem, write_coords] = btensor::op_affine_map(map_out, Ro, n, Sr, So, w.tuple(wb_idx), wb_mem2)->projs<2>();

    auto pcur = wc_mem;
    DefVec post_elems(nps);
    for (size_t j = 0; j != nps; ++j) {
        auto [pc_mem, pcoords]
            = btensor::op_affine_map(post_ins.maps[j], post_ins.rs[j], Ro, So, post_ins.ss[j], write_coords, pcur)
                  ->projs<2>();
        pcur = pc_mem;
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{pcur, mem::op_lea_tuple(k_post_dptrs[j], *Shape(pcoords).fold(post_ins.ss[j]))})
                  ->projs<2>();
        pcur          = rd_mem;
        post_elems[j] = rd_val;
    }

    auto after_post            = w.mut_con(Defs{global_ty, Tp})->set("afterPost");
    auto [post_mem, elem_post] = after_post->vars<2>();
    auto final_mem
        = w.call<mem::store>(Defs{post_mem, mem::op_lea_tuple(k_out_dptr, *Shape(write_coords).fold(So)), elem_post});
    after_post->app(true, k_ret, Defs{final_mem, k_shared, k_const, k_local});
    btensor::app_cps(write_back, global_post, {pcur, acc_final, w.tuple(post_elems)}, after_post);

    const Def* acc           = w.tuple({k_global, init});
    const Def* cont          = write_back;
    Lam* current_mut         = body;
    auto sr                  = Sr->projs(nloops_nat);
    auto raw                 = affine::counting_nest(current_mut, cont, acc, Defs(sr).subspan(ro, rr), "forRed");
    auto red_iters           = btensor::to_idxs(w, Defs(sr).subspan(ro, rr), raw);
    auto [red_mem, elem_acc] = acc->projs<2>();

    auto [body_mem, body_coords] = unflatten_index(w, flat, out_dims, red_mem);
    DefVec iters_v               = body_coords;
    iters_v.insert(iters_v.end(), red_iters.begin(), red_iters.end());
    auto iters = w.tuple(iters_v);

    auto cur = body_mem;
    DefVec input_elems(nis);
    for (size_t i = 0; i != nis; ++i) {
        auto [mc_mem, coords]
            = btensor::op_affine_map(ins.maps[i], ins.rs[i], n, Sr, ins.ss[i], iters, cur)->projs<2>();
        cur = mc_mem;
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{cur, mem::op_lea_tuple(k_dptrs[i], *Shape(coords).fold(ins.ss[i]))})->projs<2>();
        cur            = rd_mem;
        input_elems[i] = rd_val;
    }

    btensor::app_cps(current_mut, global_comb, {cur, elem_acc, w.tuple(input_elems)}, cont);

    return kernel;
}

struct TcPlan {
    nat_t B, M, N, K;
    nat_t BM, BN, BK, WY, WX, Stages;
    bool is_bf16, b_k_major;
    Vector<nat_t> batch_dims;

    Grid grid() const {
        nat_t gm = (M + BM - 1) / BM;
        nat_t gn = (N + BN - 1) / BN;
        return {B * gm * gn, WY * WX * 32, B * M * N};
    }
};

/// Whether `global_comb` is a canonical multiply-accumulate `return (m, acc + up(a) * up(b))` with zero `init`.
bool is_canonical_f32_mac(Lam* global_comb, const Def* init) {
    auto init_bits = Lit::isa<u64>(init);
    if (!init_bits || (*init_bits != 0 && *init_bits != 0x80000000ULL)) return false;

    auto body = global_comb->body()->isa<App>();
    if (!body || body->callee() != global_comb->var(2, 1)) return false;
    auto [m_var, acc_var, ins_var] = global_comb->var(2, 0)->projs<3>();
    auto [res_m, res_val]          = body->arg()->projs<2>();
    if (res_m != m_var) return false;

    auto add = Axm::isa<math::arith>(res_val);
    if (!add || add.id() != math::arith::add) return false;
    auto [a0, a1]   = add->args<2>();
    const Def* prod = nullptr;
    if (a0 == acc_var)
        prod = a1;
    else if (a1 == acc_var)
        prod = a0;
    else
        return false;

    auto mul = Axm::isa<math::arith>(prod);
    if (!mul || mul.id() != math::arith::mul) return false;
    auto peel = [](const Def* d) -> const Def* {
        if (auto cv = Axm::isa<math::conv>(d); cv && cv.id() == math::conv::f2f) return cv->arg();
        return d;
    };
    auto [p0, p1] = mul->args<2>();
    auto u0       = peel(p0);
    auto u1       = peel(p1);
    auto in0      = ins_var->proj(2, 0);
    auto in1      = ins_var->proj(2, 1);
    return (u0 == in0 && u1 == in1) || (u0 == in1 && u1 == in0);
}

std::optional<TcPlan> plan_tc_contraction(World& w,
                                          const btensor::MapRedPost& mr,
                                          nat_t nis,
                                          nat_t ro,
                                          nat_t rr,
                                          const Vector<nat_t>& out_dims,
                                          Lam* global_comb) {
    auto reject = [&](const char* why) -> std::optional<TcPlan> {
        w.log().d("gpu.pe-tc skip {}: {}", mr.Sr, why);
        return std::nullopt;
    };
    if (!w.driver().arg_nat("gpu", "pe-tc").value_or(1)) return reject("disabled");
    if (nis != 2 || ro < 2 || rr != 1) return reject("ranks");
    auto [Ta, Tb] = mr.Tis->projs<2>();
    bool is_f16   = math::isa_f(Ta) == 16 && math::isa_f(Tb) == 16;
    bool is_bf16  = math::isa_bf16(Ta) && math::isa_bf16(Tb);
    if (!is_f16 && !is_bf16) return reject("input_dtype");
    if (math::isa_f(mr.To) != 32) return reject("acc_dtype");
    if (!is_canonical_f32_mac(global_comb, mr.init)) return reject("not_mac");

    auto rn    = ro + rr;
    auto k_lit = Lit::isa<nat_t>(mr.Sr->proj(rn, ro));
    if (!k_lit || *k_lit == 0 || *k_lit % 8 != 0) return reject("k_lit");
    for (nat_t d = 0; d != ro; ++d) {
        auto so_d = Lit::isa<nat_t>(mr.So->proj(ro, d));
        if (!so_d || *so_d != out_dims[d]) return reject("so_mismatch");
    }

    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto probe  = w.mut_lam(domain, domain);
    auto vars   = probe->var()->projs(rn);
    if (w.app(mr.map_out, probe->var()) != w.tuple(Defs(vars.data(), ro))) return reject("map_out");

    auto maps  = mr.maps->projs<2>();
    auto reads = [&](size_t slot, size_t ax) {
        auto base = w.app(maps[slot], probe->var());
        auto sub  = vars;
        sub[ax]   = w.call<affine::lit>(w.lit_nat_0());
        return !Check::test_alpha(base, w.app(maps[slot], w.tuple(sub)));
    };
    auto m_ax = ro - 2;
    auto n_ax = ro - 1;
    if (reads(0, n_ax) || reads(1, m_ax)) return reject("dep");

    Vector<nat_t> batch_dims(out_dims.begin(), out_dims.begin() + m_ax);
    nat_t B = 1;
    for (auto d : batch_dims)
        B *= d;
    nat_t M = out_dims[m_ax];
    nat_t N = out_dims[n_ax];
    nat_t K = *k_lit;
    if (M < 16 || N < 16 || K < 16) return reject("too_small");

    auto is_inner_contig8 = [&](size_t slot, size_t ax, nat_t ext) -> bool {
        if (ext % 8 != 0) return false;
        auto r_lit = Lit::isa<nat_t>(mr.Ris->proj(2, slot));
        if (!r_lit || *r_lit == 0) return false;
        auto r     = *r_lit;
        auto s_in  = mr.Sis->proj(2, slot);
        auto inner = Lit::isa<nat_t>(s_in->proj(r, r - 1));
        if (!inner || *inner != ext) return false;
        auto mapped = w.app(maps[slot], probe->var());
        if (mapped->proj(r, r - 1) != vars[ax]) return false;
        auto sub = vars;
        sub[ax]  = w.call<affine::lit>(w.lit_nat_0());
        auto m0  = w.app(maps[slot], w.tuple(sub));
        for (nat_t d = 0; d + 1 < r; ++d)
            if (!Check::test_alpha(mapped->proj(r, d), m0->proj(r, d))) return false;
        return true;
    };

    if (!is_inner_contig8(0, ro, K)) return reject("a_not_k_contig8");
    bool b_n_contig = is_inner_contig8(1, n_ax, N);
    bool b_k_contig = !b_n_contig && is_inner_contig8(1, ro, K);
    if (!b_n_contig && !b_k_contig) return reject("b_not_contig8");

    nat_t bm = w.driver().arg_nat("gpu", "bm").value_or(0);
    nat_t bn = w.driver().arg_nat("gpu", "bn").value_or(0);
    nat_t bk = w.driver().arg_nat("gpu", "bk").value_or(0);
    nat_t st = w.driver().arg_nat("gpu", "tc-stages").value_or(0);

    nat_t wy, wx;
    if (bm == 0 && bn == 0) {
        if (M >= 128 && N >= 128 && K >= 32 && B * (M / 128) * (N / 128) >= 128) {
            bm = 128; bn = 128; wy = 2; wx = 4;
        } else if (M >= 64 && N >= 128 && K >= 32) {
            bm = 64; bn = 128; wy = 2; wx = 4;
        } else if (M >= 64 && N >= 64 && K >= 32) {
            bm = 64; bn = 64; wy = 2; wx = 2;
        } else {
            bm = 32; bn = 32; wy = 1; wx = 2;
        }
    } else {
        if (bm == 0) bm = M >= 128 ? 128 : (M >= 64 ? 64 : 32);
        if (bn == 0) bn = N >= 128 ? 128 : (N >= 64 ? 64 : 32);
        wy = (bm >= 64) ? 2 : 1;
        wx = (bn >= 128) ? 4 : 2;
    }
    if (bk == 0) bk = (bm >= 64 && bn >= 64 && K >= 32) ? 32 : 16;

    nat_t nt = wy * wx * 32;
    if (bm % (wy * 16) != 0 || bn % (wx * 16) != 0 || bk % 16 != 0) return reject("tile_div");
    if ((bm * (bk / 8)) % nt != 0 || (bn * (bk / 8)) % nt != 0) return reject("copy_div");

    nat_t bytes_per_stage = (bm * bk + bk * bn) * 2;
    nat_t max_stages      = std::max<nat_t>(1, 49152 / bytes_per_stage);
    nat_t k_tiles         = (K + bk - 1) / bk;
    if (st == 0) st = std::min<nat_t>({max_stages, k_tiles, 4});
    if (st == 0 || st * bytes_per_stage > 49152) return reject("smem_limit");

    return TcPlan{B, M, N, K, bm, bn, bk, wy, wx, st, is_bf16, b_k_contig, std::move(batch_dims)};
}

/// Builds a multi-stage async Tensor Core contraction kernel via `gpu.tc_tiled_kernel_m16n8k16`.
Lam* build_tc_kernel(World& w,
                     const TcPlan& plan,
                     const Def* Ro,
                     const Def* Sr,
                     const Def* So,
                     const Mapped& ins,
                     const Def* Tis,
                     const Def* To,
                     const Mapped& post_ins,
                     Lam* global_post,
                     const Def* out_dptr,
                     const Grid& grid) {
    auto nis        = ins.n();
    auto nps        = post_ins.n();
    auto ro         = plan.batch_dims.size() + 2;
    auto nloops_nat = ro + 1;
    auto n          = w.lit_nat(nloops_nat);

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();
    auto nat_ty    = w.type_nat();
    auto i64_ty    = w.type_i64();
    auto global_as = w.annex<gpu::addr_space_global>();
    auto Ta        = Tis->proj(2, 0);

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("mapReduceTcKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);

    auto m_lit  = w.lit_nat(plan.M);
    auto n_lit  = w.lit_nat(plan.N);
    auto k_lit  = w.lit_nat(plan.K);
    auto to_idx = [&](const Def* size, const Def* nat_val) {
        return w.call(core::conv::u, size, w.call<core::bitcast>(i64_ty, nat_val));
    };

    auto gptr_ty = w.app(w.annex<gpu::GlobalPtr>(), Ta);
    auto make_ptr = [&](size_t slot, bool is_a, const char* name) -> Lam* {
        auto lam = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty}, w.sigma({global_ty, gptr_ty}))->set(name);
        auto [rm, rbatch, r0, r1] = lam->vars<4>();
        auto [bm, bcoords]        = unflatten_index(w, w.call<core::bitcast>(i64_ty, rbatch), plan.batch_dims, rm);
        DefVec iters_v            = bcoords;
        if (is_a) {
            iters_v.push_back(to_idx(m_lit, r0));
            iters_v.push_back(w.lit_idx(plan.N, 0));
            iters_v.push_back(to_idx(k_lit, r1));
        } else if (plan.b_k_major) {
            iters_v.push_back(w.lit_idx(plan.M, 0));
            iters_v.push_back(to_idx(n_lit, r0));
            iters_v.push_back(to_idx(k_lit, r1));
        } else {
            iters_v.push_back(w.lit_idx(plan.M, 0));
            iters_v.push_back(to_idx(n_lit, r1));
            iters_v.push_back(to_idx(k_lit, r0));
        }
        auto [mc_mem, coords]
            = btensor::op_affine_map(ins.maps[slot], ins.rs[slot], n, Sr, ins.ss[slot], w.tuple(iters_v), bm)
                  ->projs<2>();
        auto ptr = mem::op_lea_tuple(k_dptrs[slot], *Shape(coords).fold(ins.ss[slot]));
        lam->set(true, w.tuple({mc_mem, ptr}));
        return lam;
    };

    auto ptr_a = make_ptr(0, true, "ptrA");
    auto ptr_b = make_ptr(1, false, "ptrB");

    bool post_is_id = false;
    if (nps == 0) {
        if (auto pbody = global_post->body()->isa<App>(); pbody && pbody->callee() == global_post->var(2, 1)) {
            auto [pm, pacc, _] = global_post->var(2, 0)->projs<3>();
            auto [rm, rv]      = pbody->arg()->projs<2>();
            post_is_id         = (rm == pm && rv == pacc);
        }
    }

    auto write_c = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty, To}, global_ty)->set("writeCTc");
    {
        auto [wm, wbatch, wm_nat, wn_nat, wacc] = write_c->vars<5>();
        auto [wbm, wbcoords] = unflatten_index(w, w.call<core::bitcast>(i64_ty, wbatch), plan.batch_dims, wm);
        DefVec wcoords_v     = wbcoords;
        wcoords_v.push_back(to_idx(m_lit, wm_nat));
        wcoords_v.push_back(to_idx(n_lit, wn_nat));
        auto write_coords = w.tuple(wcoords_v);
        auto pcur         = wbm;
        DefVec post_elems(nps);
        for (size_t j = 0; j != nps; ++j) {
            auto [pc_mem, pcoords]
                = btensor::op_affine_map(post_ins.maps[j], post_ins.rs[j], Ro, So, post_ins.ss[j], write_coords, pcur)
                      ->projs<2>();
            auto [rd_mem, rd_val]
                = w.call<mem::load>(Defs{pc_mem, mem::op_lea_tuple(k_post_dptrs[j], *Shape(pcoords).fold(post_ins.ss[j]))})
                      ->projs<2>();
            pcur          = rd_mem;
            post_elems[j] = rd_val;
        }
        auto [post_mem, elem_post]
            = w.app(cps::op_cps2ds_dep(global_post), w.tuple({pcur, wacc, w.tuple(post_elems)}))->projs<2>();
        auto stored
            = w.call<mem::store>(Defs{post_mem, mem::op_lea_tuple(k_out_dptr, *Shape(write_coords).fold(So)), elem_post});
        write_c->set(true, stored);
    }

    auto arr2_o   = w.arr(2, To);
    auto write_c2 = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty, arr2_o}, global_ty)->set("writeC2Tc");
    auto [wm2, wb2, wr2, wc0, wv2] = write_c2->vars<5>();
    if (post_is_id && plan.N % 2 == 0) {
        auto [wbm2, wbcoords2] = unflatten_index(w, w.call<core::bitcast>(i64_ty, wb2), plan.batch_dims, wm2);
        DefVec wcoords2        = wbcoords2;
        wcoords2.push_back(to_idx(m_lit, wr2));
        wcoords2.push_back(to_idx(n_lit, wc0));
        auto base_ptr = mem::op_lea_tuple(k_out_dptr, *Shape(w.tuple(wcoords2)).fold(So));
        auto vst      = w.app(w.app(w.annex<gpu::vstore2>(), {To, global_as}), {wbm2, base_ptr, wv2});
        write_c2->set(true, vst);
    } else {
        auto cm = w.app(write_c, Defs{wm2, wb2, wr2, wc0, wv2->proj(2, 0)});
        auto c1 = w.call(core::nat::add, Defs{wc0, w.lit_nat_1()});
        auto ok1 = w.call(core::ncmp::l, Defs{c1, n_lit});
        if (plan.N % 2 == 0) {
            cm = w.app(write_c, Defs{cm, wb2, wr2, c1, wv2->proj(2, 1)});
        } else {
            auto guard_f = w.mut_lam(global_ty, global_ty)->set("writeC2Odd");
            guard_f->set(true, w.app(write_c, Defs{guard_f->var(), wb2, wr2, c1, wv2->proj(2, 1)}));
            auto skip_f = w.mut_lam(global_ty, global_ty)->set("writeC2Skip");
            skip_f->set(true, skip_f->var());
            cm = w.app(w.extract(w.tuple({skip_f, guard_f}), ok1), cm);
        }
        write_c2->set(true, cm);
    }

    auto group_nat = grid.n_groups == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(group_id);
    auto item_nat  = grid.n_items == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(item_id);

    const Def* mma_op = plan.is_bf16 ? w.annex<gpu::mma_m16n8k16_bf16>() : w.annex<gpu::mma_m16n8k16_f16>();
    const Def* tc     = w.annex<gpu::tc_tiled_kernel_m16n8k16>();
    tc                = w.app(tc, Ta);
    tc = w.app(tc, {w.lit_nat(plan.B), m_lit, n_lit, k_lit, w.lit_nat(plan.BM), w.lit_nat(plan.BN), w.lit_nat(plan.BK),
                    w.lit_nat(plan.WY), w.lit_nat(plan.WX), w.lit_nat(plan.Stages)});
    tc = w.app(tc, w.lit_bool(plan.b_k_major));
    tc = w.app(tc, mma_op);
    tc = w.app(tc, {ptr_a, ptr_b, write_c2});
    auto [fin_global, fin_shared] = w.app(tc, {k_global, k_shared, group_nat, item_nat})->projs<2>();

    kernel->app(true, k_ret, Defs{fin_global, fin_shared, k_const, k_local});
    return kernel;
}

struct WgmmaPlan {
    nat_t M, N, K;
    nat_t BM, BN, Stages;
    bool is_bf16, b_k_major;

    Grid grid() const {
        nat_t gm = (M + BM - 1) / BM;
        nat_t gn = (N + BN - 1) / BN;
        return {gm * gn, (BM / 64) * 128, M * N};
    }
};

std::optional<WgmmaPlan> plan_wgmma_contraction(World& w,
                                                const btensor::MapRedPost& mr,
                                                nat_t nis,
                                                nat_t ro,
                                                nat_t rr,
                                                const Vector<nat_t>& out_dims,
                                                Lam* global_comb) {
    auto reject = [&](const char* why) -> std::optional<WgmmaPlan> {
        w.log().d("gpu.pe-wgmma skip {}: {}", mr.Sr, why);
        return std::nullopt;
    };
    if (!w.driver().arg_nat("gpu", "pe-wgmma").value_or(1)) return reject("disabled");
    if (nis != 2 || ro != 2 || rr != 1) return reject("ranks");

    auto [Ta, Tb] = mr.Tis->projs<2>();
    bool is_f16   = math::isa_f(Ta) == 16 && math::isa_f(Tb) == 16;
    bool is_bf16  = math::isa_bf16(Ta) && math::isa_bf16(Tb);
    if (!is_f16 && !is_bf16) return reject("input_dtype");
    if (math::isa_f(mr.To) != 32) return reject("acc_dtype");
    if (!is_canonical_f32_mac(global_comb, mr.init)) return reject("not_mac");

    auto k_lit = Lit::isa<nat_t>(mr.Sr->proj(3, 2));
    if (!k_lit || *k_lit < 64 || *k_lit % 8 != 0) return reject("k_lit");
    nat_t M = out_dims[0], N = out_dims[1], K = *k_lit;
    if (M < 128 || N < 128) return reject("too_small");
    for (nat_t d = 0; d != 2; ++d) {
        auto so_d = Lit::isa<nat_t>(mr.So->proj(2, d));
        if (!so_d || *so_d != out_dims[d]) return reject("so_mismatch");
    }

    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto probe  = w.mut_lam(domain, domain);
    auto vars   = probe->var()->projs<3>();
    if (w.app(mr.map_out, probe->var()) != w.tuple({vars[0], vars[1]})) return reject("map_out");

    auto [r_a, r_b] = mr.Ris->projs<2>();
    if (Lit::isa<nat_t>(r_a) != 2 || Lit::isa<nat_t>(r_b) != 2) return reject("in_rank");
    auto [s_a, s_b] = mr.Sis->projs<2>();
    if (Lit::isa<nat_t>(s_a->proj(2, 0)) != M || Lit::isa<nat_t>(s_a->proj(2, 1)) != K) return reject("a_shape");

    auto [map_a, map_b] = mr.maps->projs<2>();
    if (w.app(map_a, probe->var()) != w.tuple({vars[0], vars[2]})) return reject("a_map");

    auto b_mapped   = w.app(map_b, probe->var());
    bool b_nn       = (b_mapped == w.tuple({vars[2], vars[1]})) && Lit::isa<nat_t>(s_b->proj(2, 0)) == K
             && Lit::isa<nat_t>(s_b->proj(2, 1)) == N && (N % 8 == 0);
    bool b_nt       = !b_nn && (b_mapped == w.tuple({vars[1], vars[2]})) && Lit::isa<nat_t>(s_b->proj(2, 0)) == N
             && Lit::isa<nat_t>(s_b->proj(2, 1)) == K;
    if (!b_nn && !b_nt) return reject("b_map_or_shape");

    nat_t bm = w.driver().arg_nat("gpu", "bm").value_or(0);
    nat_t bn = w.driver().arg_nat("gpu", "bn").value_or(0);
    nat_t st = w.driver().arg_nat("gpu", "wgmma-stages").value_or(0);
    if (bm == 0 && bn == 0) {
        if (M >= 128 && N >= 256 && ((M + 127) / 128) * ((N + 255) / 256) >= 256) {
            bm = 128; bn = 256;
        } else if (M >= 128 && N >= 128 && ((M + 127) / 128) * ((N + 127) / 128) >= 128) {
            bm = 128; bn = 128;
        } else if (N >= 128) {
            bm = 64; bn = 128;
        } else {
            bm = 64; bn = 64;
        }
    } else {
        if (bm == 0) bm = M >= 128 ? 128 : 64;
        if (bn == 0) bn = N >= 128 ? 128 : 64;
    }
    if (bm < 64 || bm % 64 != 0 || (bm / 64) * 128 > 1024) return reject("bm");
    if (bn < 64 || bn > 256 || bn % 64 != 0) return reject("bn");

    nat_t bytes_per_stage = (bm + bn) * 128 + 16;
    nat_t max_stages      = std::max<nat_t>(1, (224 * 1024) / bytes_per_stage);
    nat_t k_tiles         = (K + 63) / 64;
    // Keep 2-3 CTAs/SM resident in H200's 228 KiB SMEM for <= 128x128 tiles; use 4 stages on 1-CTA/SM tiles.
    nat_t pref_stages     = (bm + bn <= 256) ? 3 : 4;
    if (st == 0) st = std::min<nat_t>({max_stages, k_tiles, pref_stages});
    if (st == 0 || st * bytes_per_stage > 224 * 1024) return reject("smem_limit");

    return WgmmaPlan{M, N, K, bm, bn, st, is_bf16, b_nt};
}

/// Builds a multi-stage Hopper `TMA + mbarrier + wgmma` contraction kernel via `gpu.tc_wgmma_kernel`.
Lam* build_wgmma_kernel(World& w,
                        const WgmmaPlan& plan,
                        const Def* Ro,
                        const Def* So,
                        const Def* map_a_ty,
                        const Def* map_b_ty,
                        const Def* Ta,
                        const Def* To,
                        const Mapped& post_ins,
                        Lam* global_post,
                        const Def* out_dptr,
                        const Grid& grid) {
    auto nps       = post_ins.n();
    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();
    auto nat_ty    = w.type_nat();
    auto i64_ty    = w.type_i64();
    auto global_as = w.annex<gpu::addr_space_global>();

    DefVec arg_tys(2 + nps + 1);
    arg_tys[0] = map_a_ty;
    arg_tys[1] = map_b_ty;
    for (size_t j = 0; j != nps; ++j)
        arg_tys[2 + j] = post_ins.dptrs[j]->type();
    arg_tys[2 + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("mapReduceWgmmaKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    auto k_map_a = k_args->proj(2 + nps + 1, 0);
    auto k_map_b = k_args->proj(2 + nps + 1, 1);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(2 + nps + 1, 2 + j);
    auto k_out_dptr = k_args->proj(2 + nps + 1, 2 + nps);

    auto m_lit  = w.lit_nat(plan.M);
    auto n_lit  = w.lit_nat(plan.N);
    auto k_lit  = w.lit_nat(plan.K);
    auto to_idx = [&](const Def* size, const Def* nat_val) {
        return w.call(core::conv::u, size, w.call<core::bitcast>(i64_ty, nat_val));
    };

    bool post_is_id = false;
    if (nps == 0) {
        if (auto pbody = global_post->body()->isa<App>(); pbody && pbody->callee() == global_post->var(2, 1)) {
            auto [pm, pacc, _] = global_post->var(2, 0)->projs<3>();
            auto [rm, rv]      = pbody->arg()->projs<2>();
            post_is_id         = (rm == pm && rv == pacc);
        }
    }

    auto write_c = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, To}, global_ty)->set("writeCWgmma");
    {
        auto [wm, wm_nat, wn_nat, wacc] = write_c->vars<4>();
        auto write_coords               = w.tuple({to_idx(m_lit, wm_nat), to_idx(n_lit, wn_nat)});
        auto pcur                       = wm;
        DefVec post_elems(nps);
        for (size_t j = 0; j != nps; ++j) {
            auto [pc_mem, pcoords]
                = btensor::op_affine_map(post_ins.maps[j], post_ins.rs[j], Ro, So, post_ins.ss[j], write_coords, pcur)
                      ->projs<2>();
            auto [rd_mem, rd_val]
                = w.call<mem::load>(Defs{pc_mem, mem::op_lea_tuple(k_post_dptrs[j], *Shape(pcoords).fold(post_ins.ss[j]))})
                      ->projs<2>();
            pcur          = rd_mem;
            post_elems[j] = rd_val;
        }
        auto [post_mem, elem_post]
            = w.app(cps::op_cps2ds_dep(global_post), w.tuple({pcur, wacc, w.tuple(post_elems)}))->projs<2>();
        auto stored
            = w.call<mem::store>(Defs{post_mem, mem::op_lea_tuple(k_out_dptr, *Shape(write_coords).fold(So)), elem_post});
        write_c->set(true, stored);
    }

    auto arr2_o   = w.arr(2, To);
    auto write_c2 = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, arr2_o}, global_ty)->set("writeC2Wgmma");
    auto [wm2, wr2, wc0, wv2] = write_c2->vars<4>();
    if (post_is_id && plan.N % 2 == 0) {
        auto base_ptr = mem::op_lea_tuple(k_out_dptr, *Shape(w.tuple({to_idx(m_lit, wr2), to_idx(n_lit, wc0)})).fold(So));
        auto vst      = w.app(w.app(w.annex<gpu::vstore2>(), {To, global_as}), {wm2, base_ptr, wv2});
        write_c2->set(true, vst);
    } else {
        auto cm  = w.app(write_c, Defs{wm2, wr2, wc0, wv2->proj(2, 0)});
        auto c1  = w.call(core::nat::add, Defs{wc0, w.lit_nat_1()});
        auto ok1 = w.call(core::ncmp::l, Defs{c1, n_lit});
        if (plan.N % 2 == 0) {
            cm = w.app(write_c, Defs{cm, wr2, c1, wv2->proj(2, 1)});
        } else {
            auto guard_f = w.mut_lam(global_ty, global_ty)->set("writeC2WgmmaOdd");
            guard_f->set(true, w.app(write_c, Defs{guard_f->var(), wr2, c1, wv2->proj(2, 1)}));
            auto skip_f = w.mut_lam(global_ty, global_ty)->set("writeC2WgmmaSkip");
            skip_f->set(true, skip_f->var());
            cm = w.app(w.extract(w.tuple({skip_f, guard_f}), ok1), cm);
        }
        write_c2->set(true, cm);
    }

    auto group_nat = grid.n_groups == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(group_id);
    auto item_nat  = grid.n_items == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(item_id);

    const Def* wgmma_op = plan.is_bf16 ? w.annex<gpu::wgmma_bf16>() : w.annex<gpu::wgmma_f16>();
    const Def* wg       = w.annex<gpu::tc_wgmma_kernel>();
    wg                  = w.app(wg, Ta);
    wg = w.app(wg, {m_lit, n_lit, k_lit, w.lit_nat(plan.BM), w.lit_nat(plan.BN), w.lit_nat(plan.Stages)});
    wg = w.app(wg, w.lit_bool(plan.b_k_major));
    wg = w.app(wg, wgmma_op);
    wg = w.app(wg, {k_map_a, k_map_b, write_c2});
    auto [fin_global, fin_shared] = w.app(wg, {k_global, k_shared, group_nat, item_nat})->projs<2>();

    kernel->app(true, k_ret, Defs{fin_global, fin_shared, k_const, k_local});
    return kernel;
}

struct SimtPlan {
    nat_t B, M, N, K;
    nat_t BM, BN, BK, TM, TN, Pad, Dbuf;
    bool vec4, a_v4, b_v4;
    Vector<nat_t> batch_dims;

    Grid grid() const {
        nat_t gm = (M + BM - 1) / BM;
        nat_t gn = (N + BN - 1) / BN;
        return {B * gm * gn, (BM / TM) * (BN / TN), B * M * N};
    }
};

std::optional<SimtPlan> plan_simt_contraction(World& w,
                                              const btensor::MapRedPost& mr,
                                              nat_t nis,
                                              nat_t ro,
                                              nat_t rr,
                                              const Vector<nat_t>& out_dims) {
    auto reject = [&](const char* why) -> std::optional<SimtPlan> {
        w.log().d("gpu.pe-simt skip {}: {}", mr.Sr, why);
        return std::nullopt;
    };
    if (!w.driver().arg_nat("gpu", "pe-simt").value_or(1)) return reject("disabled");
    if (nis != 2 || ro < 2 || rr != 1) return reject("ranks");
    if (mr.To == w.type_nat()) return reject("nat");

    auto rn    = ro + rr;
    auto k_lit = Lit::isa<nat_t>(mr.Sr->proj(rn, ro));
    if (!k_lit || *k_lit == 0) return reject("k_lit");
    for (nat_t d = 0; d != ro; ++d) {
        auto so_d = Lit::isa<nat_t>(mr.So->proj(ro, d));
        if (!so_d || *so_d != out_dims[d]) return reject("so_mismatch");
    }

    auto bytes_lit = Lit::isa<nat_t>(w.app(w.annex(core::trait::size), mr.To));
    if (!bytes_lit || *bytes_lit == 0) return reject("bytes");
    auto bytes = *bytes_lit;

    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto probe  = w.mut_lam(domain, domain);
    auto vars   = probe->var()->projs(rn);
    if (w.app(mr.map_out, probe->var()) != w.tuple(Defs(vars.data(), ro))) return reject("map_out");

    auto maps  = mr.maps->projs<2>();
    auto reads = [&](size_t slot, size_t ax) {
        auto base = w.app(maps[slot], probe->var());
        auto sub  = vars;
        sub[ax]   = w.call<affine::lit>(w.lit_nat_0());
        return !Check::test_alpha(base, w.app(maps[slot], w.tuple(sub)));
    };
    auto m_ax = ro - 2;
    auto n_ax = ro - 1;
    if (reads(0, n_ax) || reads(1, m_ax)) return reject("dep");

    Vector<nat_t> batch_dims(out_dims.begin(), out_dims.begin() + m_ax);
    nat_t B = 1;
    for (auto d : batch_dims)
        B *= d;
    nat_t M = out_dims[m_ax];
    nat_t N = out_dims[n_ax];
    nat_t K = *k_lit;
    // Long GEMVs (M == 1 or N == 1 with K >= 32) are better served by parallel warp reduction.
    if ((M == 1 || N == 1) && K >= 32) return reject("gemv_to_reduce");

    auto pick_b = [&](nat_t ext, nat_t cap) -> nat_t {
        if (ext >= 128 && cap >= 128) return 128;
        if (ext >= 64 && cap >= 64) return 64;
        if (ext >= 32) return 32;
        return 16;
    };
    auto pick_t = [&](nat_t b) -> nat_t {
        nat_t want = (b >= 128 && bytes <= 4) ? 8 : (b >= 32 ? 4 : 2);
        for (nat_t t = want; t >= 1; --t)
            if (b % t == 0) return t;
        return 1;
    };

    // Use 128×128 (TM=8, TN=8: 16 FMAs per LDS.128) when the grid has enough blocks to saturate SMs;
    // otherwise 64×128 (TM=4, TN=8) doubles block parallelism for medium matrices such as 1024×1024.
    nat_t bm_cap = (bytes == 4 && M >= 128 && N >= 128 && B * (M / 128) * (N / 128) >= 256) ? 128 : 64;
    nat_t bn_cap = bytes > 4 ? 64 : 128;
    nat_t bm     = w.driver().arg_nat("gpu", "bm").value_or(0);
    nat_t bn     = w.driver().arg_nat("gpu", "bn").value_or(0);
    nat_t bk     = w.driver().arg_nat("gpu", "bk").value_or(0);
    nat_t tm     = w.driver().arg_nat("gpu", "tm").value_or(0);
    nat_t tn     = w.driver().arg_nat("gpu", "tn").value_or(0);
    nat_t pad    = w.driver().arg_nat("gpu", "pad").value_or(1);
    nat_t dbuf   = w.driver().arg_nat("gpu", "dbuf").value_or(1);
    bool want_v4 = w.driver().arg_nat("gpu", "vec4").value_or(1) != 0;
    if (bm == 0) bm = pick_b(M, bm_cap);
    if (bn == 0) bn = pick_b(N, bn_cap);
    if (bk == 0) bk = K >= 16 ? 16 : 8;
    if (tm == 0) tm = pick_t(bm);
    if (tn == 0) tn = pick_t(bn);

    auto [Ta, Tb] = mr.Tis->projs<2>();
    auto ba       = Lit::isa<nat_t>(w.app(w.annex(core::trait::size), Ta));
    auto bb       = Lit::isa<nat_t>(w.app(w.annex(core::trait::size), Tb));
    nat_t nt      = (bm / tm) * (bn / tn);
    bool vec4     = want_v4 && bytes == 4 && ba == 4 && bb == 4 && (bm % 4 == 0) && (bn % 4 == 0) && (bk % 4 == 0)
             && (tm % 4 == 0) && (tn % 4 == 0) && (N % 4 == 0) && (nt > 0) && ((bm * (bk / 4)) % nt == 0)
             && ((bk * (bn / 4)) % nt == 0);

    auto is_inner_contig = [&](size_t slot, size_t ax, nat_t ext) -> bool {
        if (ext % 4 != 0) return false;
        auto r_lit = Lit::isa<nat_t>(mr.Ris->proj(2, slot));
        if (!r_lit || *r_lit == 0) return false;
        auto r     = *r_lit;
        auto s_in  = mr.Sis->proj(2, slot);
        auto inner = Lit::isa<nat_t>(s_in->proj(r, r - 1));
        if (!inner || *inner != ext) return false;
        auto mapped = w.app(maps[slot], probe->var());
        if (mapped->proj(r, r - 1) != vars[ax]) return false;
        auto sub = vars;
        sub[ax]  = w.call<affine::lit>(w.lit_nat_0());
        auto m0  = w.app(maps[slot], w.tuple(sub));
        for (nat_t d = 0; d + 1 < r; ++d)
            if (!Check::test_alpha(mapped->proj(r, d), m0->proj(r, d))) return false;
        return true;
    };

    bool a_v4 = vec4 && is_inner_contig(0, ro, K);
    bool b_v4 = vec4 && is_inner_contig(1, n_ax, N);

    return SimtPlan{B, M, N, K, bm, bn, bk, tm, tn, pad, dbuf, vec4, a_v4, b_v4, std::move(batch_dims)};
}

/// Builds a shared-memory + register-tiled SIMT contraction kernel via `gpu.simt_tiled_kernel`.
Lam* build_simt_kernel(World& w,
                       const SimtPlan& plan,
                       const Def* Ro,
                       const Def* Sr,
                       const Def* So,
                       const Mapped& ins,
                       const Def* Tis,
                       const Def* To,
                       const Def* init,
                       Lam* global_comb,
                       const Mapped& post_ins,
                       Lam* global_post,
                       const Def* out_dptr,
                       const Grid& grid) {
    auto nis        = ins.n();
    auto nps        = post_ins.n();
    auto ro         = plan.batch_dims.size() + 2;
    auto nloops_nat = ro + 1;
    auto n          = w.lit_nat(nloops_nat);

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();
    auto nat_ty    = w.type_nat();
    auto i64_ty    = w.type_i64();
    auto [Ta, Tb]  = Tis->projs<2>();

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("mapReduceSimtKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);

    auto m_lit = w.lit_nat(plan.M);
    auto n_lit = w.lit_nat(plan.N);
    auto k_lit = w.lit_nat(plan.K);

    auto to_idx = [&](const Def* size, const Def* nat_val) {
        return w.call(core::conv::u, size, w.call<core::bitcast>(i64_ty, nat_val));
    };

    auto make_reader = [&](size_t slot, const Def* elem_ty, bool is_a, const char* name) -> Lam* {
        auto lam = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty}, w.sigma({global_ty, elem_ty}))->set(name);
        auto [rm, rbatch, r0, r1] = lam->vars<4>();
        auto [bm, bcoords]        = unflatten_index(w, w.call<core::bitcast>(i64_ty, rbatch), plan.batch_dims, rm);
        DefVec iters_v            = bcoords;
        if (is_a) {
            iters_v.push_back(to_idx(m_lit, r0));
            iters_v.push_back(w.lit_idx(plan.N, 0));
            iters_v.push_back(to_idx(k_lit, r1));
        } else {
            iters_v.push_back(w.lit_idx(plan.M, 0));
            iters_v.push_back(to_idx(n_lit, r1));
            iters_v.push_back(to_idx(k_lit, r0));
        }
        auto [mc_mem, coords]
            = btensor::op_affine_map(ins.maps[slot], ins.rs[slot], n, Sr, ins.ss[slot], w.tuple(iters_v), bm)
                  ->projs<2>();
        auto ld = w.call<mem::load>(Defs{mc_mem, mem::op_lea_tuple(k_dptrs[slot], *Shape(coords).fold(ins.ss[slot]))});
        lam->set(true, ld);
        return lam;
    };

    auto read_a = make_reader(0, Ta, true, "readA");
    auto read_b = make_reader(1, Tb, false, "readB");

    auto write_c = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty, To}, global_ty)->set("writeC");
    auto [wm, wbatch, wm_nat, wn_nat, wacc] = write_c->vars<5>();
    auto [wbm, wbcoords] = unflatten_index(w, w.call<core::bitcast>(i64_ty, wbatch), plan.batch_dims, wm);
    DefVec wcoords_v     = wbcoords;
    wcoords_v.push_back(to_idx(m_lit, wm_nat));
    wcoords_v.push_back(to_idx(n_lit, wn_nat));
    auto write_coords = w.tuple(wcoords_v);

    auto pcur = wbm;
    DefVec post_elems(nps);
    for (size_t j = 0; j != nps; ++j) {
        auto [pc_mem, pcoords]
            = btensor::op_affine_map(post_ins.maps[j], post_ins.rs[j], Ro, So, post_ins.ss[j], write_coords, pcur)
                  ->projs<2>();
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{pc_mem, mem::op_lea_tuple(k_post_dptrs[j], *Shape(pcoords).fold(post_ins.ss[j]))})
                  ->projs<2>();
        pcur          = rd_mem;
        post_elems[j] = rd_val;
    }
    auto [post_mem, elem_post]
        = w.app(cps::op_cps2ds_dep(global_post), w.tuple({pcur, wacc, w.tuple(post_elems)}))->projs<2>();
    auto stored
        = w.call<mem::store>(Defs{post_mem, mem::op_lea_tuple(k_out_dptr, *Shape(write_coords).fold(So)), elem_post});
    write_c->set(true, stored);

    auto group_nat = grid.n_groups == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(group_id);
    auto item_nat  = grid.n_items == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(item_id);

    const Def* fin_pair;
    if (plan.vec4) {
        auto global_as = w.annex<gpu::addr_space_global>();
        auto make_reader4 = [&](size_t slot, const Def* elem_ty, bool is_a, bool contig, Lam* scalar_rd,
                                const char* name) -> Lam* {
            auto arr4 = w.arr(4, elem_ty);
            auto lam  = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty}, w.sigma({global_ty, arr4}))->set(name);
            auto [rm, rbatch, r0, r1] = lam->vars<4>();
            if (contig) {
                auto [bm, bcoords] = unflatten_index(w, w.call<core::bitcast>(i64_ty, rbatch), plan.batch_dims, rm);
                DefVec iters_v     = bcoords;
                if (is_a) {
                    iters_v.push_back(to_idx(m_lit, r0));
                    iters_v.push_back(w.lit_idx(plan.N, 0));
                    iters_v.push_back(to_idx(k_lit, r1));
                } else {
                    iters_v.push_back(w.lit_idx(plan.M, 0));
                    iters_v.push_back(to_idx(n_lit, r1));
                    iters_v.push_back(to_idx(k_lit, r0));
                }
                auto [mc_mem, coords]
                    = btensor::op_affine_map(ins.maps[slot], ins.rs[slot], n, Sr, ins.ss[slot], w.tuple(iters_v), bm)
                          ->projs<2>();
                auto base_ptr = mem::op_lea_tuple(k_dptrs[slot], *Shape(coords).fold(ins.ss[slot]));
                auto vld      = w.app(w.app(w.annex<gpu::vload4>(), {elem_ty, global_as}), {mc_mem, base_ptr});
                lam->set(true, vld);
            } else {
                const Def* cm = rm;
                DefVec vals(4);
                for (nat_t l = 0; l != 4; ++l) {
                    auto c1        = l == 0 ? r1 : w.call(core::nat::add, Defs{r1, w.lit_nat(l)});
                    auto [nm, val] = w.app(scalar_rd, Defs{cm, rbatch, r0, c1})->projs<2>();
                    cm             = nm;
                    vals[l]        = val;
                }
                lam->set(true, w.tuple({cm, w.tuple(vals)}));
            }
            return lam;
        };

        auto read_a4 = make_reader4(0, Ta, true, plan.a_v4, read_a, "readA4");
        auto read_b4 = make_reader4(1, Tb, false, plan.b_v4, read_b, "readB4");

        // Check whether global_post is the identity `return (m, acc)` so we can emit direct `vstore4`.
        bool post_is_id = false;
        if (nps == 0) {
            if (auto pbody = global_post->body()->isa<App>(); pbody && pbody->callee() == global_post->var(2, 1)) {
                auto [pm, pacc, _] = global_post->var(2, 0)->projs<3>();
                auto [rm, rv]      = pbody->arg()->projs<2>();
                post_is_id         = (rm == pm && rv == pacc);
            }
        }

        auto arr4_o   = w.arr(4, To);
        auto write_c4 = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, nat_ty, arr4_o}, global_ty)->set("writeC4");
        auto [wm4, wb4, wr4, wc0, wv4] = write_c4->vars<5>();
        if (post_is_id) {
            auto [wbm4, wbcoords4] = unflatten_index(w, w.call<core::bitcast>(i64_ty, wb4), plan.batch_dims, wm4);
            DefVec wcoords4        = wbcoords4;
            wcoords4.push_back(to_idx(m_lit, wr4));
            wcoords4.push_back(to_idx(n_lit, wc0));
            auto base_ptr = mem::op_lea_tuple(k_out_dptr, *Shape(w.tuple(wcoords4)).fold(So));
            auto vst      = w.app(w.app(w.annex<gpu::vstore4>(), {To, global_as}), {wbm4, base_ptr, wv4});
            write_c4->set(true, vst);
        } else {
            const Def* cm = wm4;
            for (nat_t l = 0; l != 4; ++l) {
                auto col = l == 0 ? wc0 : w.call(core::nat::add, Defs{wc0, w.lit_nat(l)});
                cm       = w.app(write_c, Defs{cm, wb4, wr4, col, wv4->proj(4, l)});
            }
            write_c4->set(true, cm);
        }

        const Def* simt = w.annex<gpu::simt_tiled_kernel_v4>();
        simt            = w.app(simt, {Ta, Tb, To});
        simt = w.app(simt, {w.lit_nat(plan.B), m_lit, n_lit, k_lit, w.lit_nat(plan.BM), w.lit_nat(plan.BN),
                            w.lit_nat(plan.BK), w.lit_nat(plan.TM), w.lit_nat(plan.TN), w.lit_nat(plan.Pad),
                            w.lit_nat(plan.Dbuf)});
        simt = w.app(simt, {init, global_comb});
        simt = w.app(simt, {read_a4, read_b4, read_a, read_b, write_c4});
        fin_pair = w.app(simt, {k_global, k_shared, group_nat, item_nat});
    } else {
        const Def* simt = w.annex<gpu::simt_tiled_kernel>();
        simt            = w.app(simt, {Ta, Tb, To});
        simt = w.app(simt, {w.lit_nat(plan.B), m_lit, n_lit, k_lit, w.lit_nat(plan.BM), w.lit_nat(plan.BN),
                            w.lit_nat(plan.BK), w.lit_nat(plan.TM), w.lit_nat(plan.TN), w.lit_nat(plan.Pad)});
        simt     = w.app(simt, {init, global_comb});
        simt     = w.app(simt, {read_a, read_b, write_c});
        fin_pair = w.app(simt, {k_global, k_shared, group_nat, item_nat});
    }
    auto [fin_global, fin_shared] = fin_pair->projs<2>();

    kernel->app(true, k_ret, Defs{fin_global, fin_shared, k_const, k_local});
    return kernel;
}

bool mentions_def(const Def* root, const Def* target, DefSet& seen) {
    if (root == target) return true;
    if (auto [_, ins] = seen.emplace(root); !ins) return false;
    for (auto d : root->deps())
        if (mentions_def(d, target, seen)) return true;
    return false;
}

/// Extracts an associative binary `merge: Fn [GlobalM, To, To] → [GlobalM, To]` from a pure combiner
/// `comb(m, acc, ins) = return (m, op(acc, rhs))` whenever `init` is a valid neutral/idempotent seed.
Lam* extract_merge_fn(World& w, Lam* global_comb, const Def* To, const Def* init) {
    auto body = global_comb->body()->isa<App>();
    if (!body || body->callee() != global_comb->var(2, 1)) return nullptr;
    auto [m_var, acc_var, _] = global_comb->var(2, 0)->projs<3>();
    auto [res_m, res_val]    = body->arg()->projs<2>();
    if (res_m != m_var) return nullptr;

    auto op_app = res_val->isa<App>();
    if (!op_app || op_app->arg()->num_projs() != 2) return nullptr;
    auto [lhs, rhs] = op_app->arg()->projs<2>();
    const Def* other = nullptr;
    if (lhs == acc_var)
        other = rhs;
    else if (rhs == acc_var)
        other = lhs;
    else
        return nullptr;

    DefSet seen;
    if (mentions_def(other, acc_var, seen)) return nullptr;

    auto init_bits = Lit::isa<u64>(init);
    auto bytes_lit = Lit::isa<nat_t>(w.app(w.annex(core::trait::size), To));
    if (!bytes_lit) return nullptr;
    auto bytes = *bytes_lit;

    auto is_float_zero = [&]() {
        if (!init_bits) return false;
        if (*init_bits == 0) return true;
        if (bytes == 4) return *init_bits == 0x80000000ULL;
        if (bytes == 8) return *init_bits == 0x8000000000000000ULL;
        return false;
    };
    auto is_float_one = [&]() {
        if (!init_bits) return false;
        if (bytes == 4) return *init_bits == 0x3f800000ULL;
        if (bytes == 8) return *init_bits == 0x3ff0000000000000ULL;
        return false;
    };

    bool ok = false;
    if (auto ma = Axm::isa<math::arith>(op_app)) {
        ok = (ma.id() == math::arith::add && is_float_zero()) || (ma.id() == math::arith::mul && is_float_one());
    } else if (Axm::isa<math::extrema>(op_app)) {
        ok = true; // min/max are associative, commutative and idempotent for any init.
    } else if (auto cw = Axm::isa<core::wrap>(op_app)) {
        ok = (cw.id() == core::wrap::add && init_bits == 0) || (cw.id() == core::wrap::mul && init_bits == 1);
    } else if (Axm::isa<core::extrema>(op_app)) {
        ok = true;
    } else if (auto b2 = Axm::isa<core::bit2>(op_app)) {
        ok = b2.id() == core::bit2::and_ || b2.id() == core::bit2::or_
          || (b2.id() == core::bit2::xor_ && init_bits == 0);
    }
    if (!ok) return nullptr;

    auto global_ty     = w.annex<gpu::GlobalM>();
    auto merge         = w.mut_con(Defs{w.sigma({global_ty, To, To}), w.cn({global_ty, To})})->set("mergeGlobal");
    auto [mm, ma, mb]  = merge->var(2, 0)->projs<3>();
    merge->app(true, merge->var(2, 1), Defs{mm, w.app(op_app->callee(), Defs{ma, mb})});
    return merge;
}

struct SimtReducePlan {
    nat_t OutN, RedK, NT;
    Vector<nat_t> red_dims;
    Lam* merge;

    Grid grid() const { return {OutN, NT, OutN}; }
};

std::optional<SimtReducePlan> plan_simt_reduction(World& w,
                                                  const btensor::MapRedPost& mr,
                                                  nat_t ro,
                                                  nat_t rr,
                                                  const Vector<nat_t>& out_dims,
                                                  Lam* global_comb) {
    if (!w.driver().arg_nat("gpu", "pe-simt").value_or(1)) return std::nullopt;
    if (rr == 0) return std::nullopt;
    if (mr.To == w.type_nat() || mr.To->isa<Sigma>() || mr.To->isa<Arr>()) return std::nullopt;

    auto bytes_lit = Lit::isa<nat_t>(w.app(w.annex(core::trait::size), mr.To));
    if (!bytes_lit || (*bytes_lit != 1 && *bytes_lit != 2 && *bytes_lit != 4 && *bytes_lit != 8)) return std::nullopt;

    auto rn = ro + rr;
    Vector<nat_t> red_dims(rr);
    nat_t RedK = 1;
    for (nat_t j = 0; j != rr; ++j) {
        auto l = Lit::isa<nat_t>(mr.Sr->proj(rn, ro + j));
        if (!l || *l == 0) return std::nullopt;
        red_dims[j] = *l;
        RedK *= *l;
    }

    nat_t nt_override = w.driver().arg_nat("gpu", "reduce-nt").value_or(0);
    if (nt_override == 0 && RedK < 32) return std::nullopt;

    auto merge = extract_merge_fn(w, global_comb, mr.To, mr.init);
    if (!merge) return std::nullopt;

    nat_t OutN = 1;
    for (auto d : out_dims)
        OutN *= d;

    nat_t NT = nt_override;
    if (NT == 0) {
        nat_t max_nt = OutN <= 16 ? 512 : 256;
        NT           = 32;
        while (NT * 2 <= max_nt && NT * 2 <= RedK)
            NT *= 2;
    }

    return SimtReducePlan{OutN, RedK, NT, std::move(red_dims), merge};
}

/// Builds a warp-shuffled parallel reduction kernel via `gpu.simt_reduce_kernel`.
Lam* build_simt_reduce_kernel(World& w,
                              const SimtReducePlan& plan,
                              const Def* Ro,
                              const Vector<nat_t>& out_dims,
                              const Def* Sr,
                              const Def* So,
                              const Mapped& ins,
                              const Def* To,
                              const Def* map_out,
                              const Def* init,
                              Lam* global_comb,
                              const Mapped& post_ins,
                              Lam* global_post,
                              const Def* out_dptr,
                              const Grid& grid) {
    auto nis        = ins.n();
    auto nps        = post_ins.n();
    auto ro         = out_dims.size();
    auto rr         = plan.red_dims.size();
    auto nloops_nat = ro + rr;
    auto n          = w.lit_nat(nloops_nat);

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();
    auto nat_ty    = w.type_nat();
    auto i64_ty    = w.type_i64();

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("mapReduceSimtReduceKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);

    auto step_elem = w.mut_lam(Defs{global_ty, nat_ty, nat_ty, To}, w.sigma({global_ty, To}))->set("stepElem");
    auto [sm, sout_nat, sred_nat, sacc] = step_elem->vars<4>();
    auto [om, ocoords] = unflatten_index(w, w.call<core::bitcast>(i64_ty, sout_nat), out_dims, sm);
    auto [rm, rcoords] = unflatten_index(w, w.call<core::bitcast>(i64_ty, sred_nat), plan.red_dims, om);
    DefVec iters_v     = ocoords;
    iters_v.insert(iters_v.end(), rcoords.begin(), rcoords.end());
    auto iters = w.tuple(iters_v);

    auto cur = rm;
    DefVec input_elems(nis);
    for (size_t i = 0; i != nis; ++i) {
        auto [mc_mem, coords]
            = btensor::op_affine_map(ins.maps[i], ins.rs[i], n, Sr, ins.ss[i], iters, cur)->projs<2>();
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{mc_mem, mem::op_lea_tuple(k_dptrs[i], *Shape(coords).fold(ins.ss[i]))})
                  ->projs<2>();
        cur            = rd_mem;
        input_elems[i] = rd_val;
    }
    auto comb_res = w.app(cps::op_cps2ds_dep(global_comb), w.tuple({cur, sacc, w.tuple(input_elems)}));
    step_elem->set(true, comb_res);

    auto write_out = w.mut_lam(Defs{global_ty, nat_ty, To}, global_ty)->set("writeOut");
    auto [wm, wout_nat, wval] = write_out->vars<3>();
    auto [wom, wocoords]      = unflatten_index(w, w.call<core::bitcast>(i64_ty, wout_nat), out_dims, wm);
    DefVec wb_idx             = wocoords;
    for (size_t j = 0; j != rr; ++j)
        wb_idx.push_back(w.call(core::conv::u, Sr->proj(nloops_nat, ro + j), w.lit_i64(0)));
    auto [wc_mem, write_coords] = btensor::op_affine_map(map_out, Ro, n, Sr, So, w.tuple(wb_idx), wom)->projs<2>();

    auto pcur = wc_mem;
    DefVec post_elems(nps);
    for (size_t j = 0; j != nps; ++j) {
        auto [pc_mem, pcoords]
            = btensor::op_affine_map(post_ins.maps[j], post_ins.rs[j], Ro, So, post_ins.ss[j], write_coords, pcur)
                  ->projs<2>();
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{pc_mem, mem::op_lea_tuple(k_post_dptrs[j], *Shape(pcoords).fold(post_ins.ss[j]))})
                  ->projs<2>();
        pcur          = rd_mem;
        post_elems[j] = rd_val;
    }
    auto [post_mem, elem_post]
        = w.app(cps::op_cps2ds_dep(global_post), w.tuple({pcur, wval, w.tuple(post_elems)}))->projs<2>();
    auto stored
        = w.call<mem::store>(Defs{post_mem, mem::op_lea_tuple(k_out_dptr, *Shape(write_coords).fold(So)), elem_post});
    write_out->set(true, stored);

    auto group_nat = grid.n_groups == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(group_id);
    auto item_nat  = grid.n_items == 1 ? w.lit_nat_0() : w.call<core::idx2nat>(item_id);

    const Def* red = w.annex<gpu::simt_reduce_kernel>();
    red            = w.app(red, To);
    red            = w.app(red, {w.lit_nat(plan.OutN), w.lit_nat(plan.RedK), w.lit_nat(plan.NT)});
    red            = w.app(red, {init, plan.merge});
    red            = w.app(red, {step_elem, write_out});
    auto [fin_global, fin_shared] = w.app(red, {k_global, k_shared, group_nat, item_nat})->projs<2>();

    kernel->app(true, k_ret, Defs{fin_global, fin_shared, k_const, k_local});
    return kernel;
}

Lam* build_teardown(World& w,
                    const Def* Ro,
                    const Def* So,
                    const Def* Tp,
                    Defs dptrs,
                    Defs post_dptrs,
                    const Def* out_dptr,
                    const Def* host_buf,
                    const Def* cont) {
    auto global_ty = w.annex<gpu::GlobalM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto mem_ty    = w.call<mem::M>(0);

    auto after_launch                        = w.mut_con(Defs{mem_ty, global_ty, const_ty})->set("afterLaunch");
    auto [post_mem, post_global, post_const] = after_launch->vars<3>();

    auto copy_back
        = w.app(w.app(w.annex<gpu::buf_copy_to_host>(), {Ro, So, Tp}), {post_mem, post_global, out_dptr, host_buf});
    auto [cb_mem, cb_global] = copy_back->projs<2>();

    auto cur_global = cb_global;
    for (auto dptr : dptrs)
        cur_global = w.call(gpu::free::block, Defs{cur_global, dptr});
    for (auto dptr : post_dptrs)
        cur_global = w.call(gpu::free::block, Defs{cur_global, dptr});
    cur_global = w.call(gpu::free::block, Defs{cur_global, out_dptr});

    auto final_mem = w.app(w.annex<gpu::auto_deinit>(), Defs{cb_mem, cur_global, post_const});
    after_launch->app(true, cont, Defs{final_mem, host_buf});
    return after_launch;
}

} // namespace

void LowerMapReduce::start() {
    DefSet seen;
    auto has_gpu_init
        = std::ranges::any_of(old_world().externals().muts(), [&](auto def) { return contains_gpu_init(def, seen); });
    if (has_gpu_init) {
        log().w("not lowering any map-reduce operations to GPU: the program already contains an explicit `gpu.init`");
        return;
    }
    Super::start();
}

const Def* LowerMapReduce::rewrite_imm_App(const App* app) {
    if (Axm::isa<btensor::map_reduce_post>(app)) return lower_map_reduce_post(app);
    return Super::rewrite_imm_App(app);
}

const Def* LowerMapReduce::lower_map_reduce_post(const App* app) {
    auto& w        = new_world();
    auto mr        = btensor::MapRedPost::of_btensor(rewrite(app->callee())->as<App>());
    auto Sr        = mr.Sr;
    auto result_ty = rewrite(app->type());

    auto ranks = mr.ranks();
    if (!ranks) {
        log().w("{} doesn't have lowering-time known rank counts (nis/nps/Ro/Rn)", app);
        return Super::rewrite_imm_App(app);
    }
    if (!Check::test_alpha(mr.init_type, mr.To)) {
        log().w("{} seeds its cells from memory, which the GPU lowering does not support", app);
        return Super::rewrite_imm_App(app);
    }
    auto [nis_n, nps_n, ro, rr] = *ranks;

    Vector<nat_t> out_dims(ro);
    nat_t out_total = 1;
    for (nat_t d = 0; d != ro; ++d) {
        auto l = Lit::isa<nat_t>(Sr->proj(ro + rr, d));
        if (!l) {
            log().w("{} doesn't have a lowering-time known output (grid) shape", app);
            return Super::rewrite_imm_App(app);
        }
        out_dims[d] = *l;
        out_total *= *l;
    }
    if (out_total == 0) {
        log().w("{} has a zero-sized output, skipping GPU lowering", app);
        return Super::rewrite_imm_App(app);
    }

    auto comb_lam = mr.comb->isa_mut<Lam>();
    auto post_lam = mr.post->isa_mut<Lam>();
    if (!comb_lam || !post_lam) {
        log().w("{} doesn't have a lowering-time known combiner/epilogue", app);
        return Super::rewrite_imm_App(app);
    }

    auto [fun, call] = btensor::ds_splice(rewrite(app->arg()), result_ty, "mapReduceAffGpu");
    auto [fun_mem, host_buf, new_inputs, new_post_ins] = fun->var(2, 0)->projs<4>();
    auto cont                                          = fun->var(2, 1);

    auto [h_mem, h_global, h_const] = w.app(w.annex<gpu::auto_init>(), fun_mem)->projs<3>();

    auto in_desc = extract_input_desc(nis_n, mr.Ris, mr.Sis, mr.Tis, mr.maps);
    auto inputs  = alloc_copy_inputs(w, h_mem, h_global, in_desc.rs, in_desc.ss, in_desc.ts, new_inputs);

    auto post_desc = extract_input_desc(nps_n, mr.Rps, mr.Sps, mr.Tps, mr.post_maps);
    auto post_inputs
        = alloc_copy_inputs(w, inputs.mem, inputs.global, post_desc.rs, post_desc.ss, post_desc.ts, new_post_ins);

    auto [out_global, out_dptr] = alloc_output(w, post_inputs.global, mr.Tp, mr.So);

    auto global_comb = rebuild_lam_global_mem(comb_lam, mr.To, w.sym("combGlobal"));
    auto global_post = rebuild_lam_global_mem(post_lam, mr.Tp, w.sym("postGlobal"));

    Grid grid;
    Lam* kernel;
    DefVec kernel_in_args = inputs.dptrs;
    DefVec free_dptrs     = inputs.dptrs;
    if (auto wg_plan = plan_wgmma_contraction(w, mr, nis_n, ro, rr, out_dims, global_comb)) {
        log().i("tile the {} contraction over {} on GPU with gpu.tc_wgmma_kernel (BM={}, BN={}, BK=64, Stages={}, b_k_major={})",
                wg_plan->is_bf16 ? "bf16" : "f16", Sr, wg_plan->BM, wg_plan->BN, wg_plan->Stages,
                wg_plan->b_k_major ? 1 : 0);
        auto Ta      = mr.Tis->proj(2, 0);
        auto a_point = Axm::expect<mem::Ptr>(inputs.dptrs[0]->type(), "a `mem.Ptr`")->arg(2, 0);
        auto b_point = Axm::expect<mem::Ptr>(inputs.dptrs[1]->type(), "a `mem.Ptr`")->arg(2, 0);
        auto [g1, map_a]
            = w.app(w.app(w.app(w.annex<gpu::device_tensor_map>(), Ta), a_point),
                    {out_global, inputs.dptrs[0], w.lit_nat(wg_plan->M), w.lit_nat(wg_plan->K), w.lit_nat(wg_plan->BM),
                     w.lit_nat(64)})
                  ->projs<2>();
        nat_t b_rows     = wg_plan->b_k_major ? wg_plan->N : wg_plan->K;
        nat_t b_cols     = wg_plan->b_k_major ? wg_plan->K : wg_plan->N;
        nat_t b_box_rows = wg_plan->b_k_major ? wg_plan->BN : 64;
        auto [g2, map_b]
            = w.app(w.app(w.app(w.annex<gpu::device_tensor_map>(), Ta), b_point),
                    {g1, inputs.dptrs[1], w.lit_nat(b_rows), w.lit_nat(b_cols), w.lit_nat(b_box_rows), w.lit_nat(64)})
                  ->projs<2>();
        out_global     = g2;
        kernel_in_args = {map_a, map_b};
        free_dptrs.push_back(map_a);
        free_dptrs.push_back(map_b);
        grid   = wg_plan->grid();
        kernel = build_wgmma_kernel(w, *wg_plan, mr.Ro, mr.So, map_a->type(), map_b->type(), Ta, mr.To,
                                    Mapped{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.maps}, global_post,
                                    out_dptr, grid);
    } else if (auto tc_plan = plan_tc_contraction(w, mr, nis_n, ro, rr, out_dims, global_comb)) {
        log().i("tile the {} contraction over {} on GPU with gpu.tc_tiled_kernel_m16n8k16 (BM={}, BN={}, BK={}, WY={}, WX={}, Stages={}, b_k_major={})",
                tc_plan->is_bf16 ? "bf16" : "f16", Sr, tc_plan->BM, tc_plan->BN, tc_plan->BK, tc_plan->WY, tc_plan->WX,
                tc_plan->Stages, tc_plan->b_k_major ? 1 : 0);
        grid   = tc_plan->grid();
        kernel = build_tc_kernel(w, *tc_plan, mr.Ro, Sr, mr.So,
                                 Mapped{in_desc.rs, in_desc.ss, inputs.dptrs, in_desc.maps}, mr.Tis, mr.To,
                                 Mapped{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.maps}, global_post,
                                 out_dptr, grid);
    } else if (auto plan = plan_simt_contraction(w, mr, nis_n, ro, rr, out_dims)) {
        log().i("tile the contraction over {} on GPU with gpu.simt_tiled_kernel{} (BM={}, BN={}, BK={}, TM={}, TN={}, dbuf={})",
                Sr, plan->vec4 ? "_v4" : "", plan->BM, plan->BN, plan->BK, plan->TM, plan->TN, plan->Dbuf);
        grid   = plan->grid();
        kernel = build_simt_kernel(w, *plan, mr.Ro, Sr, mr.So,
                                   Mapped{in_desc.rs, in_desc.ss, inputs.dptrs, in_desc.maps}, mr.Tis, mr.To, mr.init,
                                   global_comb, Mapped{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.maps},
                                   global_post, out_dptr, grid);
    } else if (auto rplan = plan_simt_reduction(w, mr, ro, rr, out_dims, global_comb)) {
        log().i("reduce {} on GPU with gpu.simt_reduce_kernel (OutN={}, RedK={}, NT={})", Sr, rplan->OutN, rplan->RedK,
                rplan->NT);
        grid   = rplan->grid();
        kernel = build_simt_reduce_kernel(w, *rplan, mr.Ro, out_dims, Sr, mr.So,
                                          Mapped{in_desc.rs, in_desc.ss, inputs.dptrs, in_desc.maps}, mr.To, mr.map_out,
                                          mr.init, global_comb,
                                          Mapped{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.maps},
                                          global_post, out_dptr, grid);
    } else {
        grid   = grid_layout(out_dims);
        kernel = build_kernel(w, mr.Ro, rr, out_dims, Sr, mr.So,
                              Mapped{in_desc.rs, in_desc.ss, inputs.dptrs, in_desc.maps}, mr.To, mr.map_out, mr.init,
                              global_comb, Mapped{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.maps},
                              global_post, mr.Tp, out_dptr, grid);
    }

    DefVec kernel_args = kernel_in_args;
    kernel_args.insert(kernel_args.end(), post_inputs.dptrs.begin(), post_inputs.dptrs.end());
    kernel_args.push_back(out_dptr);
    DefVec kernel_arg_tys(kernel_args.size(), [&](size_t i) { return kernel_args[i]->type(); });

    auto launch = w.app(w.annex<gpu::launch>(), Defs{w.lit_nat(kernel_args.size()), w.tuple(kernel_arg_tys)});
    launch      = w.app(launch, Defs{w.lit_nat(grid.n_groups), w.lit_nat(grid.n_items), w.annex<gpu::default_stream>(),
                                     w.lit_ff(), w.tuple()});
    launch      = w.app(launch, kernel);
    launch      = w.app(launch, kernel_args);

    auto after_launch
        = build_teardown(w, mr.Ro, mr.So, mr.Tp, free_dptrs, post_inputs.dptrs, out_dptr, host_buf, cont);
    auto launch_call = w.app(launch, Defs{w.tuple({post_inputs.mem, out_global, h_const}), after_launch});
    fun->set(true, launch_call);

    return call;
}

} // namespace mim::plug::gpu::phase
