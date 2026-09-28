#include "mim/plug/btensor/phase/lower_map_reduce.h"

#include <mim/axm.h>
#include <mim/def.h>
#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/mem/mem.h>

#include <algorithm>

#include <fe/worklist.h>

#include "mim/plug/btensor/btensor.h"

namespace mim::plug::btensor::phase {

namespace {

/// Builds a counting `affine.For` loop body carrying `acc` (a `{mem, …}` tuple).
std::pair<Lam*, const Def*> counting_for(const Def* bound, const Def* acc, const Def* exit, Sym name) {
    auto& w       = bound->world();
    auto acc_ty   = acc->type();
    auto body     = w.mut_con({/* iter */ w.type_i64(), /* acc */ acc_ty, /* return */ w.cn(acc_ty)})->set(name);
    auto for_loop = w.call<affine::For>(body, exit, Defs{w.lit_i64(0), bound, w.lit_i64(1), acc});
    return {body, for_loop};
}

/// Pointwise scaffold shared by `pad`/`concat`: a `[mem, ins] → [mem, Buf]` fun (spliced via
/// `cps.cps2ds`) that allocates the output buffer, loops over `s_out` carrying `{mem, buf}`, and writes
/// `compute`'s elem at the (identity) output coordinates.
/// `compute(iters, ins, mem)` receives the raw i64 loop counters, the fun's inputs var, and the current
/// mem; it returns `(mem', elem)`.
const Def* build_pointwise(World& w,
                           const Def* result_ty, // [mem.M 0, buffer.Buf (r, s_out, T)]
                           const Def* op_mem,
                           const Def* op_ins,
                           const Def* s_out,
                           u64 rn,
                           const std::string& name,
                           auto compute) {
    auto mem_ty         = w.call<mem::M>(0);
    auto fun            = w.mut_fun(w.sigma({mem_ty, op_ins->type()}), result_ty)->set(name);
    auto call           = w.app(cps::op_cps2ds_dep(fun), w.tuple({op_mem, op_ins}));
    auto [fun_mem, ins] = fun->var(2, 0)->projs<2>();
    auto cont           = fun->var(2, 1);

    auto [obr, obs, obT]  = Axm::isa<buffer::Buf>(result_ty->proj(2, 1))->args<3>();
    auto [a_mem, out_buf] = buffer::op_alloc(obr, obs, obT, fun_mem)->projs<2>();
    const Def* acc        = w.tuple({a_mem, out_buf});
    auto current_mut      = fun;

    DefVec iters; // raw i64 loop counters
    iters.reserve(rn);
    for (u64 d = 0; d < rn; ++d) {
        auto bound                  = w.call<core::bitcast>(w.type_i64(), s_out->proj(rn, d));
        auto [body, for_call]       = counting_for(bound, acc, cont, w.sym(name + "_" + std::to_string(d)));
        auto [iter, new_acc, yield] = body->vars<3>();
        cont                        = yield;
        iters.push_back(iter);
        acc = new_acc;
        current_mut->set(true, for_call);
        current_mut = body;
    }
    auto [loop_mem, loop_buf] = acc->projs<2>();

    std::pair<const Def*, const Def*> el = compute(iters, ins, loop_mem);
    auto [el_mem, elem]                  = el;

    DefVec wcoords(rn);
    for (u64 d = 0; d < rn; ++d)
        wcoords[d] = w.call(core::conv::u, s_out->proj(rn, d), iters[d]);
    auto [wr_mem, wr_buf]
        = buffer::op_write(obr, obs, obT, el_mem, loop_buf, *Shape(w, wcoords).fold(s_out), elem)->projs<2>();
    current_mut->app(true, cont, w.tuple({wr_mem, loop_buf}));
    return call;
}

} // namespace

const Def* LowerMapReduce::rewrite_imm_App(const App* app) {
    if (is_bootstrapping()) return RWPhase::rewrite_imm_App(app);
    if (Axm::isa<btensor::map_reduce_post>(app)) return lower_map_reduce_post(app);
    if (Axm::isa<btensor::compute_at>(app)) return rewrite(app->arg());
    if (Axm::isa<btensor::broadcast>(app)) return lower_broadcast(app);
    if (Axm::isa<btensor::pad>(app)) return lower_pad(app);
    if (Axm::isa<btensor::concat>(app)) return lower_concat(app);
    if (Axm::isa<buffer::lit>(app)) return lower_buffer_lit(app);
    if (Axm::isa<btensor::gather>(app)) return lower_gather(app);
    if (Axm::isa<btensor::scatter>(app)) return lower_scatter(app);
    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerMapReduce::lower_buffer_lit(const App* app) {
    // `buffer.lit (r, s, T) (mem, val)` fills every elem with `val`. Emit a fill loop (via the same
    // pointwise scaffold as pad/concat) so the store never materializes as one giant literal array. A
    // non-literal rank has no static loop nest, so leave it for `buffer.lower_ptr`'s monolithic fallback.
    auto [r, s, T] = app->callee()->as<App>()->args<3>();
    auto s_out     = rewrite(s);
    auto rn        = Lit::isa<u64>(rewrite(r));
    if (!rn) return RWPhase::rewrite_imm_App(app);

    auto& w         = new_world();
    auto [mem, val] = rewrite(app->arg())->projs<2>();
    auto result_ty  = rewrite(app->type()); // [mem.M 0, buffer.Buf (r, s, T)]
    // `compute` ignores the loop counters and writes the (loop-invariant) scalar `ins` everywhere.
    return build_pointwise(
        w, result_ty, mem, val, s_out, *rn, "constant_fill",
        [](const DefVec&, const Def* ins, const Def* m) -> std::pair<const Def*, const Def*> { return {m, ins}; });
}

/// The pieces of a (new-world) `btensor.map_reduce_post` callee.
struct MrOp {
    u64 nis, nps, ro, rr;
    const Def *To, *Tp, *Ro, *Sr, *So, *sched, *Ris, *Sis, *Rps, *Sps, *comb, *init, *post, *accs, *post_accs, *acc_out;
};

namespace {

std::optional<MrOp> mr_op(const App* c) {
    auto [nis_nps, meta, shapes, in_tys, comb_init, acc_out, accs_all] = c->uncurry_args<7>();
    auto [nis, nps]                                                    = nis_nps->projs<2>();
    auto [To, Tp, Ro, Rn, TSched]                                      = meta->projs<5>();
    auto [So, Sr, sched]                                               = shapes->projs<3>();
    auto [Tis, Ris, Sis, Tps, Rps, Sps]                                = in_tys->projs<6>();
    auto [comb, init, post]                                            = comb_init->projs<3>();
    auto [accs, post_accs]                                             = accs_all->projs<2>();
    auto nis_l = Lit::isa<u64>(nis), nps_l = Lit::isa<u64>(nps), ro_l = Lit::isa<u64>(Ro), rn_l = Lit::isa<u64>(Rn);
    if (!nis_l || !nps_l || !ro_l || !rn_l || *rn_l < *ro_l) return {};
    return MrOp{*nis_l, *nps_l, *ro_l, *rn_l - *ro_l, To, Tp, Ro, Sr, So, sched, Ris, Sis, Rps, Sps,
                comb, init, post, accs, post_accs, acc_out};
}

/// The coordinate `e` of an access map over the loop vector `var` of `n` dims as `cst + Σ coef[d] · var#d`.
std::optional<LowerMapReduce::Lin> linear(const Def* var, u64 n, const Def* e) {
    auto lin = LowerMapReduce::Lin{std::vector<s64>(n, 0), 0};
    if (n == 1 && e == var) return lin.coef[0] = 1, lin;
    if (auto ex = e->isa<Extract>(); ex && ex->tuple() == var) {
        if (auto i = Lit::isa<u64>(ex->index()); i && *i < n) return lin.coef[*i] = 1, lin;
        return {};
    }
    if (auto l = Axm::isa<affine::lit>(e)) {
        if (auto c = Lit::isa<u64>(l->arg())) return lin.cst = s64(*c), lin;
        return {};
    }
    if (auto mul = Axm::isa(affine::semiop::mul, e)) {
        auto [x, c] = mul->args<2>();
        auto k      = Lit::isa<u64>(c);
        auto lx     = k ? linear(var, n, x) : std::nullopt;
        if (!lx) return {};
        for (auto& a : lx->coef)
            a *= s64(*k);
        lx->cst *= s64(*k);
        return lx;
    }
    if (auto op = Axm::isa<affine::op>(e); op && (op.id() == affine::op::add || op.id() == affine::op::sub)) {
        auto [a, b] = op->args<2>();
        auto la = linear(var, n, a), lb = linear(var, n, b);
        if (!la || !lb) return {};
        s64 sign = op.id() == affine::op::add ? 1 : -1;
        for (u64 d = 0; d != n; ++d)
            la->coef[d] += sign * lb->coef[d];
        la->cst += sign * lb->cst;
        return la;
    }
    return {};
}

/// The coordinates of the access map `map` (a lam over «n; affine.Index») as linear forms.
std::optional<std::vector<LowerMapReduce::Lin>> linear_map(const Def* map, u64 n, u64 r) {
    auto lam = map->isa_mut<Lam>();
    if (!lam || !lam->is_set()) return {};
    auto var  = lam->var();
    auto body = lam->body();
    if (body == var && n == r) {
        auto res = std::vector<LowerMapReduce::Lin>(r, LowerMapReduce::Lin{std::vector<s64>(n, 0), 0});
        for (u64 a = 0; a != r; ++a)
            res[a].coef[a] = 1;
        return res;
    }
    auto res  = std::vector<LowerMapReduce::Lin>();
    for (u64 a = 0; a != r; ++a) {
        auto e = r == 1 ? body : body->isa<Tuple>() ? body->op(a) : nullptr;
        if (!e) return {};
        auto l = linear(var, n, e);
        if (!l) return {};
        res.emplace_back(std::move(*l));
    }
    return res;
}

std::optional<std::vector<u64>> literals(const Def* s, u64 n) {
    auto res = std::vector<u64>(n);
    for (u64 i = 0; i != n; ++i)
        if (auto l = Lit::isa<u64>(s->proj(n, i)))
            res[i] = *l;
        else
            return {};
    return res;
}

} // namespace

void LowerMapReduce::start() {
    // Uses of every def, charged through the tuples and packs that merely wrap arguments.
    auto uses   = DefMap<u64>();
    auto charge = [&](this auto&& charge, const Def* d) -> void {
        ++uses[d];
        if (d->isa<Tuple>() || d->isa<Pack>())
            for (auto op : d->ops())
                if (op) charge(op);
    };
    auto mrs = std::vector<const App*>();
    auto wl  = fe::BFSWorklist<DefSet>();
    wl.push(old_world().roots());
    while (!wl.empty()) {
        auto def = wl.pop();
        if (auto mr = Axm::isa<btensor::map_reduce_post>(def)) mrs.emplace_back(mr);
        auto transparent = def->isa<Tuple>() || def->isa<Pack>();
        for (auto op : def->ops())
            if (op) {
                if (!transparent) charge(op);
                wl.push(op);
            }
        if (def->type()) wl.push(def->type());
    }

    for (auto consumer : mrs) {
        auto c = mr_op(consumer->callee()->as<App>());
        if (!c) continue;
        auto is = consumer->arg()->proj(3, 1);
        for (u64 i = 0; i != c->nis; ++i) {
            auto at = Axm::isa<btensor::compute_at>(is->proj(c->nis, i));
            if (!at || uses[at] != 1) continue;
            auto ex = at->arg()->isa<Extract>();
            if (!ex || uses[ex] != 1 || Lit::isa<u64>(ex->index()) != 1) continue;
            auto producer = Axm::isa<btensor::map_reduce_post>(ex->tuple());
            if (!producer) continue;
            auto p     = mr_op(producer->callee()->as<App>());
            auto level = Lit::isa<u64>(at->callee()->as<App>()->arg());
            if (!p || !level || *level > c->ro) continue;

            auto n_c  = c->ro + c->rr;
            auto sr_c = literals(c->Sr, n_c);
            auto so_p = literals(p->So, p->ro);
            auto lin  = linear_map(c->accs->proj(c->nis, i), n_c, p->ro);
            auto out  = linear_map(p->acc_out, p->ro + p->rr, p->ro);
            if (!sr_c || !so_p || !lin || !out) continue;

            // The producer writes output axis a from its parallel loop perm[a]: a projection.
            auto stage = Stage{producer, *level, *lin, std::vector<u64>(p->ro), std::vector<u64>(p->ro)};
            bool ok    = true;
            for (u64 a = 0; ok && a != p->ro; ++a) {
                auto& o = (*out)[a];
                auto it = std::ranges::find(o.coef, 1);
                ok      = o.cst == 0 && std::ranges::count(o.coef, 0) == s64(o.coef.size()) - 1 && it != o.coef.end()
                  && u64(it - o.coef.begin()) < p->ro;
                if (ok) stage.perm[a] = it - o.coef.begin();
                // The tile spans what the loops inside the leading `level` dims read of axis a.
                u64 ext = 1;
                for (u64 d = 0; ok && d != n_c; ++d) {
                    ok &= stage.lin[a].coef[d] >= 0;
                    if (d >= *level) ext += u64(stage.lin[a].coef[d]) * ((*sr_c)[d] - 1);
                }
                ok &= stage.lin[a].cst >= 0 && ext <= (*so_p)[a];
                stage.ext[a] = ext;
            }
            if (!ok) continue;
            stages_.emplace(at, std::move(stage));
            consumers_.emplace(at, consumer);
        }
    }
    // A staged producer is lowered as a plain nest in its consumer's stage, which does not stage operands itself.
    for (auto& [at, stage] : stages_)
        dropped_.insert(stage.producer);
    for (auto it = stages_.begin(); it != stages_.end();)
        if (dropped_.contains(consumers_[it->first]))
            it = stages_.erase(it);
        else
            ++it;
    dropped_.clear();
    for (auto& [at, stage] : stages_) {
        log().d("compute {} at level {} of {}", stage.producer, stage.level, consumers_[at]);
        dropped_.insert(stage.producer);
    }
    RWPhase::start();
}

const Def* LowerMapReduce::build_nest(const MrOp& op,
                                      const Def* Sr_loop,
                                      const Def* U,
                                      const DefVec& in_bufs,
                                      const DefVec& in_maps,
                                      const DefVec& in_shapes,
                                      const Def* post_bufs,
                                      const DefVec& shift,
                                      const Tile* tile,
                                      const Def* sl,
                                      std::function<Lam*(const Def*)> make_stage) {
    auto& w     = new_world();
    auto nloops = op.ro + op.rr;
    auto n      = w.lit_nat(nloops);
    auto i32    = w.type_i32();

    // Builds `affine.map @(m, n) @(sin, sout) f idxs mem`. The map is mem-threaded (its divisions consume mem),
    // and this phase threads real memory, so the caller passes the current mem and receives `(mem', coords)`.
    auto affine_map = [&](const Def* f, const Def* m, const Def* nn, const Def* sin, const Def* sout, const Def* idxs,
                          const Def* mem) {
        auto a = w.app(w.annex<affine::map>(), w.tuple({m, nn}));
        a      = w.app(a, w.tuple({sin, sout}));
        a      = w.app(a, f);
        a      = w.app(a, idxs);
        a      = w.app(a, w.lit_nat_0());
        return w.app(a, mem)->projs<2>();
    };

    // A combiner/epilogue operand canonically has the axm's `Fn` shape `Cn [[args], Cn ret]`, but an earlier
    // Scalarize may have flattened an escaped lam to `Cn [args…, Cn ret]` — build the argument to match the
    // callee's actual domain either way.
    auto apply_cps = [&](Lam* mut, const Def* f, DefVec parts, const Def* k) {
        auto dom = f->type()->as<Pi>()->dom();
        if (dom->num_projs() == parts.size() + 1) {
            parts.emplace_back(k);
            mut->app(true, f, w.tuple(parts));
        } else {
            mut->app(true, f, w.tuple({w.tuple(parts), k}));
        }
    };

    // The nest value's loop-vector components may appear as one «r; I32» value or flattened into r
    // separate I32 components (normalization decides) — index the domains verbatim either way.
    auto load_ivs = [&](Lam* l, u64 ndom, u64 pos, u64 cnt) {
        DefVec out(cnt);
        if (ndom == pos + cnt + 1) // flattened: cnt I32 scalars, then the continuation
            for (u64 d = 0; d < cnt; ++d)
                out[d] = l->var(ndom, pos + d);
        else
            for (u64 d = 0; d < cnt; ++d)
                out[d] = l->var(ndom, pos)->proj(cnt, d);
        return out;
    };

    // Loop dim d of the domain `Sr_loop` at `iv`, shifted into the op's own domain `op.Sr`.
    auto at_dim = [&](u64 d, const Def* iv) {
        if (!shift.empty() && shift[d]) iv = w.call(core::wrap::add, core::Mode::nsuw, Defs{iv, shift[d]});
        return w.call(core::conv::u, op.Sr->proj(nloops, d), iv);
    };

    // The op's SCHEDULE `sched` is a target-agnostic chooser over a loop-nest builder (canonically
    // a `tensor.mk_sched` value, selected in the frontend). This is the tensor→btensor boundary,
    // so bind it HERE to this target's algebra — `btensor.mr_nest` over the write-back target — then
    // build only the decision-free pieces: the fold step `cell` (read one element per input, call
    // the combiner) and the write-back `wb` (read the epilogue inputs, run `post`, store) — and
    // APPLY the nest to them. Unrolling, interchange and the row accumulator are inside the
    // builder: plain IR, not lowering behavior.
    auto nest_args = w.tuple({op.Ro, w.lit_nat(op.rr), Sr_loop, op.To, U});
    auto nest      = w.app(w.app(op.sched, w.app(w.annex<btensor::NestT>(), nest_args)),
                           w.app(w.annex<btensor::mr_nest>(), nest_args));

    // The bound nest dictates the exact `cell`/`wb` signatures (its [init, cell, wb, sl, stage] domain) —
    // building them from the VALUE's own type sidesteps any Arr/Sigma normalization asymmetry.
    auto sched_dom = nest->type()->as<Pi>()->dom();

    // cell: Cn [mem, To, «ro+rr; I32», Cn [mem, To]] — fold the elements at one loop vector.
    auto cdom = sched_dom->proj(5, 1)->as<Pi>()->dom();
    auto cn   = cdom->num_projs();
    auto cell = w.mut_con(cdom)->set("cell");
    {
        auto cm   = cell->var(cn, 0);
        auto cacc = cell->var(cn, 1);
        auto ck   = cell->var(cn, cn - 1);
        auto civs = load_ivs(cell, cn, 2, nloops);
        DefVec iters_v(nloops, [&](size_t d) { return at_dim(d, civs[d]); });
        auto iters = w.tuple(iters_v);
        auto cur   = cm;
        DefVec input_elems(op.nis);
        for (u64 i = 0; i < op.nis; ++i) {
            auto [mc_mem, coords]
                = affine_map(in_maps[i], op.Ris->proj(op.nis, i), n, op.Sr, in_shapes[i], iters, cur);
            cur                   = mc_mem;
            auto folded           = *Shape(coords).fold(in_shapes[i]);
            auto [ir, is_, iT]    = Axm::isa<buffer::Buf>(in_bufs[i]->type())->args<3>();
            auto [rd_mem, rd_val] = buffer::op_read(ir, is_, iT, cur, in_bufs[i], folded)->projs<2>();
            cur                   = rd_mem;
            input_elems[i]        = rd_val;
        }
        apply_cps(cell, op.comb, {cur, cacc, w.tuple(input_elems)}, ck);
    }

    // wb: Cn [mem, U, To, «ro+rr; I32», Cn [mem, U]] — epilogue + store for one folded cell,
    // threading the write-back target through the nest. It receives the full loop
    // vector; only the leading `ro` output coordinates are read (the trailing reduction slots are
    // exhausted loop values and are replaced by zeros for `acc_out`).
    auto wdom = sched_dom->proj(5, 2)->as<Pi>()->dom();
    auto wn   = wdom->num_projs();
    auto wb   = w.mut_con(wdom)->set("wb");
    {
        auto wm   = wb->var(wn, 0);
        auto wu   = wb->var(wn, 1);
        auto wv   = wb->var(wn, 2);
        auto wk   = wb->var(wn, wn - 1);
        auto wovs = load_ivs(wb, wn, 3, nloops);
        DefVec wb_iters(nloops);
        for (u64 i = 0; i < op.ro; ++i)
            wb_iters[i] = at_dim(i, wovs[i]);
        for (u64 j = 0; j < op.rr; ++j)
            wb_iters[op.ro + j] = w.call(core::conv::u, op.Sr->proj(nloops, op.ro + j), w.lit(i32, 0));
        auto [wc_mem, write_coords] = affine_map(op.acc_out, op.Ro, n, op.Sr, op.So, w.tuple(wb_iters), wm);

        auto pcur = wc_mem;
        DefVec post_elems(op.nps);
        for (u64 j = 0; j < op.nps; ++j) {
            auto sps_j             = op.Sps->proj(op.nps, j);
            auto [pc_mem, pcoords] = affine_map(op.post_accs->proj(op.nps, j), op.Rps->proj(op.nps, j), op.Ro, op.So,
                                                sps_j, write_coords, pcur);
            pcur                   = pc_mem;
            auto p_buf             = post_bufs->proj(op.nps, j);
            auto [pr, ps_, pT]     = Axm::isa<buffer::Buf>(p_buf->type())->args<3>();
            auto [prd_mem, p_val] = buffer::op_read(pr, ps_, pT, pcur, p_buf, *Shape(pcoords).fold(sps_j))->projs<2>();
            pcur                  = prd_mem;
            post_elems[j]         = p_val;
        }
        auto after_post            = mem::mut_con(op.Tp)->set("afterPost");
        auto [post_mem, elem_post] = after_post->vars<2>();
        auto [ur, us, uT]          = Axm::isa<buffer::Buf>(U)->args<3>();
        auto target                = *Shape(write_coords).fold(op.So);
        if (tile) {
            // A tile is indexed by the unshifted loop values of the parallel dims that write each axis.
            auto r = tile->perm.size();
            DefVec tc(r, [&](size_t a) { return w.call(core::conv::u, tile->ext->proj(r, a), wovs[tile->perm[a]]); });
            target = *Shape(w.tuple(tc)).fold(tile->ext);
        }
        auto stored = buffer::op_write(ur, us, uT, post_mem, wu, target, elem_post);
        after_post->app(true, wk, w.tuple({stored->proj(2, 0), stored->proj(2, 1)}));
        apply_cps(wb, op.post, {pcur, wv, w.tuple(post_elems)}, after_post);
    }

    // stage: Cn [«ro+rr; I32», mem, Cn mem] — a staged operand's producer, or nothing.
    auto sdom  = sched_dom->proj(5, 4)->as<Pi>()->dom();
    Lam* stage = make_stage ? make_stage(sdom) : nullptr;
    if (!stage) {
        stage = w.mut_con(sdom)->set("noStage");
        stage->app(true, stage->var(3, 2), stage->var(3, 1));
    }
    return w.app(nest, w.tuple({op.init, cell, wb, sl ? sl : w.lit_nat_0(), stage}));
}

const Def* LowerMapReduce::lower_map_reduce_post(const App* app) {
    auto& w = new_world();
    // A producer computed inside its consumer: its mem passes through, its buffer is never read.
    if (dropped_.contains(app)) {
        auto ty = rewrite(app->type());
        return w.tuple({rewrite(app->arg()->proj(3, 0)), w.bot(ty->proj(2, 1))});
    }

    auto c  = rewrite(app->callee())->as<App>();
    auto op = mr_op(c);
    if (!op) {
        log().w("rank counts (nis/nps/Ro/Rn) of {} are not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }

    // The final argument is `[mem, is, post_is]`; the result is `[mem, Buf]`.
    auto [op_mem, op_is, op_post_is] = rewrite(app->arg())->projs<3>();
    auto result_ty                   = rewrite(app->type()); // [mem.M 0, buffer.Buf (Ro, So, Tp)]
    auto mem_ty                      = w.call<mem::M>(0);

    // `[mem, is, post_is] → [mem, Buf]`, spliced via cps.cps2ds and applied to the op's (mem, is, post_is).
    auto fun  = w.mut_fun(w.sigma({mem_ty, op_is->type(), op_post_is->type()}), result_ty)->set("mapRedAff");
    auto call = w.app(cps::op_cps2ds_dep(fun), w.tuple({op_mem, op_is, op_post_is}));
    auto [fun_mem, new_inputs, new_post_is] = fun->var(2, 0)->projs<3>();
    auto cont                               = fun->var(2, 1);

    // Allocate the output buffer.
    auto out_ty           = result_ty->proj(2, 1);
    auto [obr, obs, obT]  = Axm::isa<buffer::Buf>(out_ty)->args<3>();
    auto [a_mem, out_buf] = buffer::op_alloc(obr, obs, obT, fun_mem)->projs<2>();

    DefVec in_bufs(op->nis, [&](size_t i) { return new_inputs->proj(op->nis, i); });
    DefVec in_maps(op->nis, [&](size_t i) { return op->accs->proj(op->nis, i); });
    DefVec in_shapes(op->nis, [&](size_t i) { return op->Sis->proj(op->nis, i); });

    // At most one staged (`compute_at`) operand: its producer runs in `stage` into a tile the operand then reads.
    const Def* sl    = nullptr;
    auto old_is      = app->arg()->proj(3, 1);
    auto nloops      = op->ro + op->rr;
    const Stage* stg = nullptr;
    u64 slot         = 0;
    for (u64 i = 0; i != op->nis && !stg; ++i)
        if (auto it = stages_.find(old_is->proj(op->nis, i)); it != stages_.end()) stg = &it->second, slot = i;

    const Def* mem = a_mem;
    Tile tile;
    const Def* tile_ty  = nullptr;
    const Def* tile_buf = nullptr;
    if (stg) {
        auto pc = rewrite(stg->producer->callee())->as<App>();
        auto p  = *mr_op(pc);
        auto r  = p.ro;
        DefVec ext(r, [&](size_t a) { return w.lit_nat(stg->ext[a]); });
        auto ext_t    = w.tuple(ext);
        tile          = Tile{stg->perm, ext_t};
        tile_ty       = w.call<buffer::Buf>(Defs{p.Ro, ext_t, p.Tp});
        auto alloc    = buffer::op_alloc(p.Ro, ext_t, p.Tp, mem)->projs<2>();
        mem           = alloc[0];
        tile_buf      = alloc[1];
        in_bufs[slot] = tile_buf;
        in_shapes[slot] = ext_t;

        // The operand reads the tile at its coordinate minus the tile's origin: the part of each linear form over
        // the loops inside the leading `level` dims.
        auto vec_ty = op->accs->proj(op->nis, slot)->type()->as<Pi>()->dom();
        auto rd     = w.mut_lam(vec_ty, w.arr(p.Ro, w.annex<affine::Index>()))->set("tile_map");
        DefVec coords(r, [&](size_t a) {
            const Def* e = w.call<affine::lit>(w.lit_nat_0());
            for (u64 d = stg->level; d != nloops; ++d)
                if (auto k = stg->lin[a].coef[d])
                    e = w.call(affine::op::add,
                               Defs{e, w.call(affine::semiop::mul, Defs{rd->var(nloops, d), w.lit_nat(u64(k))})});
            return e;
        });
        rd->set(true, w.tuple(coords));
        in_maps[slot] = rd;
        sl            = w.lit_nat(stg->level);
    }

    // The producer's nest over the tile, shifted to the tile's origin, which the enclosing loop values determine.
    auto make_stage = [&](const Def* sdom) -> Lam* {
        auto stage = w.mut_con(sdom)->set("stage");
        auto sn    = sdom->num_projs();
        auto pc    = rewrite(stg->producer->callee())->as<App>();
        auto p     = *mr_op(pc);
        auto pa    = rewrite(stg->producer->arg());
        auto p_is  = pa->proj(3, 1);
        auto p_ps  = pa->proj(3, 2);
        auto k     = stage->var(sn, sn - 1);
        auto pm    = stage->var(sn, sn - 2);
        // The loop vector: one «r; I32» value or flattened I32 scalars.
        DefVec pref(nloops);
        for (u64 d = 0; d != nloops; ++d)
            pref[d] = sn == 3 ? stage->var(sn, 0)->proj(nloops, d) : stage->var(sn, d);

        auto i32 = w.type_i32();
        auto pn  = p.ro + p.rr;
        DefVec shift(pn, nullptr);
        DefVec sr_t(pn, [&](size_t d) { return p.Sr->proj(pn, d); });
        for (u64 a = 0; a != p.ro; ++a) {
            const Def* lo = w.lit(i32, u64(stg->lin[a].cst));
            for (u64 d = 0; d != stg->level; ++d)
                if (auto kk = stg->lin[a].coef[d])
                    lo = w.call(core::wrap::add, core::Mode::nsuw,
                                Defs{lo, w.call(core::wrap::mul, core::Mode::nsuw, Defs{pref[d], w.lit(i32, u64(kk))})});
            shift[stg->perm[a]] = lo;
            sr_t[stg->perm[a]]  = w.lit_nat(stg->ext[a]);
        }
        DefVec p_bufs(p.nis, [&](size_t i) { return p_is->proj(p.nis, i); });
        DefVec p_maps(p.nis, [&](size_t i) { return p.accs->proj(p.nis, i); });
        DefVec p_shapes(p.nis, [&](size_t i) { return p.Sis->proj(p.nis, i); });
        auto p_nest = build_nest(p, w.tuple(sr_t), tile_ty, p_bufs, p_maps, p_shapes, p_ps, shift, &tile, nullptr, {});
        auto done   = w.mut_con({mem_ty, tile_ty})->set("staged");
        done->app(true, k, done->var(2, 0));
        stage->app(true, p_nest, w.tuple({pm, tile_buf, done}));
        return stage;
    };
    auto nest = build_nest(*op, op->Sr, out_ty, in_bufs, in_maps, in_shapes, new_post_is, {}, nullptr, sl,
                           stg ? std::function<Lam*(const Def*)>(make_stage) : nullptr);

    // Apply the nest; the output buffer is threaded through as the nest's write-back target and
    // yielded straight to the op's continuation.
    fun->app(true, nest, w.tuple({mem, out_buf, cont}));
    return call;
}

const Def* LowerMapReduce::lower_broadcast(const App* app) {
    auto& w              = new_world();
    auto callee          = app->callee()->as<App>(); // (broadcast {impl}) (s_in, s_out)
    auto [s_in, s_out]   = rewrite(callee->arg())->projs<2>();
    auto [op_mem, input] = rewrite(app->arg())->projs<2>();
    auto result_ty       = rewrite(app->type()); // [mem.M 0, buffer.Buf (ro, so, T)]

    auto r_nat = s_out->num_projs();

    auto mem_ty            = w.call<mem::M>(0);
    auto fun               = w.mut_fun(w.sigma({mem_ty, input->type()}), result_ty)->set("broadcast");
    auto call              = w.app(cps::op_cps2ds_dep(fun), w.tuple({op_mem, input}));
    auto [fun_mem, in_buf] = fun->var(2, 0)->projs<2>();
    auto cont              = fun->var(2, 1);

    auto [in_r, in_s, in_T]    = Axm::isa<buffer::Buf>(in_buf->type())->args<3>();
    auto [out_r, out_s, out_T] = Axm::isa<buffer::Buf>(result_ty->proj(2, 1))->args<3>();

    auto [a_mem, out_buf] = buffer::op_alloc(out_r, out_s, out_T, fun_mem)->projs<2>();
    const Def* acc        = w.tuple({a_mem, out_buf});
    auto current_mut      = fun;
    DefVec out_iters;
    out_iters.reserve(r_nat);
    for (size_t i = 0; i < r_nat; ++i) {
        auto dim                    = s_out->proj(r_nat, i);
        auto bound                  = w.call<core::bitcast>(w.type_i64(), dim);
        auto [body, for_call]       = counting_for(bound, acc, cont, w.sym("bcast_" + std::to_string(i)));
        auto [iter, new_acc, yield] = body->vars<3>();
        cont                        = yield;
        out_iters.push_back(w.call(core::conv::u, dim, iter));
        acc = new_acc;
        current_mut->set(true, for_call);
        current_mut = body;
    }
    auto [loop_mem, loop_buf] = acc->projs<2>();

    // Non-size-1 input dims mirror the matching output index; size-1 dims are dropped from each buffer index.
    auto iters            = w.tuple(out_iters);
    auto iter_fold_out    = *Shape(iters).fold(s_out);
    auto iter_fold_in     = *Shape(iters).fold(s_in);
    auto [rd_mem, rd_val] = buffer::op_read(in_r, in_s, in_T, loop_mem, in_buf, iter_fold_in)->projs<2>();
    auto [wr_mem, wr_buf] = buffer::op_write(out_r, out_s, out_T, rd_mem, loop_buf, iter_fold_out, rd_val)->projs<2>();
    current_mut->app(true, cont, w.tuple({wr_mem, loop_buf}));

    return call;
}

const Def* LowerMapReduce::lower_pad(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    // callee: pad {T, r} [s_in] [mode, lo, hi] [s_out]. The shapes are the logical ones; buffer reads and
    // writes fold size-1 axes (the `Buf` handles are normalized), while the loops cover all logical dims.
    auto [Tr, s_in, params, s_out] = c->uncurry_args<4>();
    auto [mode, lo, hi]            = params->projs<3>();
    auto [op_mem, input, val]      = rewrite(app->arg())->projs<3>();
    auto result_ty                 = rewrite(app->type()); // [mem.M 0, buffer.Buf (r, s_out, T)]

    auto r_l    = Lit::isa<u64>(Tr->proj(2, 1));
    auto mode_l = Lit::isa<u64>(mode);
    if (!r_l || !mode_l) {
        log().w("rank/mode of {} is not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }
    auto rn       = *r_l;
    auto mode_nat = *mode_l;
    auto i64      = w.type_i64();

    // select(cond, t, f) == `(f, t)#cond` (cf. core.select); cond : Bool.
    auto sel = [&](const Def* cond, const Def* t, const Def* f) { return w.extract(w.tuple({f, t}), cond); };

    auto compute = [&](const DefVec& iters, const Def* ins, const Def* mem) -> std::pair<const Def*, const Def*> {
        auto [in_buf, fill]  = ins->projs<2>();
        auto [ibr, ibs, ibT] = Axm::isa<buffer::Buf>(in_buf->type())->args<3>();
        DefVec clamped(rn); // per-axis read index, kept in range, as `Idx (s_in#d)`
        DefVec valid;       // per-axis in-bounds flag (constant mode only)
        for (u64 d = 0; d < rn; ++d) {
            auto lo_d  = w.call<core::bitcast>(i64, lo->proj(rn, d));
            auto sin_d = w.call<core::bitcast>(i64, s_in->proj(rn, d));
            auto in_d  = w.call(core::wrap::sub, core::Mode::none, Defs{iters[d], lo_d}); // o#d − lo#d
            const Def* idx_i64;
            if (mode_nat == 0) { // constant: a single unsigned `<` covers both bounds (underflow wraps high)
                auto v_d = w.call(core::icmp::ul, w.tuple({in_d, sin_d}));
                valid.push_back(v_d);
                idx_i64 = sel(v_d, in_d, w.lit_i64(0));
            } else { // replicate: clamp the read to the nearest edge [0, s_in#d − 1]
                auto sin_m1 = w.call(core::wrap::sub, core::Mode::none, Defs{sin_d, w.lit_i64(1)});
                idx_i64     = w.call(core::extrema::smax,
                                     w.tuple({w.lit_i64(0), w.call(core::extrema::smin, w.tuple({in_d, sin_m1}))}));
            }
            clamped[d] = w.call(core::conv::u, s_in->proj(rn, d), idx_i64);
        }
        auto [rd_mem, elem] = buffer::op_read(ibr, ibs, ibT, mem, in_buf, *Shape(w, clamped).fold(s_in))->projs<2>();
        if (mode_nat != 0) return {rd_mem, elem}; // replicate: always a (clamped) read
        auto all_valid = valid.empty() ? w.lit_tt() : valid[0];
        for (u64 d = 1; d < valid.size(); ++d)
            all_valid = w.call(core::bit2::and_, w.lit_nat(2), w.tuple({all_valid, valid[d]}));
        return {rd_mem, sel(all_valid, elem, fill)}; // constant: fill out-of-region cells with `val`
    };

    return build_pointwise(w, result_ty, op_mem, w.tuple({input, val}), s_out, rn, "pad", compute);
}

const Def* LowerMapReduce::lower_concat(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    // callee: concat {T, nis, r} [ax] {Sis} [s_out]. The shapes are the logical ones; buffer reads and
    // writes fold size-1 axes (the `Buf` handles are normalized), while the loops cover all logical dims.
    auto [TnisR, ax, Sis, s_out] = c->uncurry_args<4>();
    auto [T, nis, r]             = TnisR->projs<3>();
    auto [op_mem, op_is]         = rewrite(app->arg())->projs<2>();
    auto result_ty               = rewrite(app->type()); // [mem.M 0, buffer.Buf (r, s_out, T)]

    auto nis_l = Lit::isa<u64>(nis);
    auto r_l   = Lit::isa<u64>(r);
    auto ax_l  = Lit::isa<u64>(ax);
    if (!nis_l || !r_l || !ax_l) {
        log().w("nis/rank/axis of {} are not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }
    auto nisn = *nis_l, rn = *r_l, axn = *ax_l;

    // Prefix offsets along `ax`: off#i = Σ_{j<i} Sis#i#ax (literal extents required).
    DefVec off(nisn);
    fe::Vector<u64> ext(nisn);
    u64 acc_off = 0;
    for (u64 i = 0; i < nisn; ++i) {
        off[i]  = w.lit_i64(acc_off);
        auto ei = Lit::isa<u64>(Sis->proj(nisn, i)->proj(rn, axn));
        if (!ei) {
            log().w("extent of input {} of {} along the concat axis is not known at lowering time", i, app);
            return RWPhase::rewrite_imm_App(app);
        }
        ext[i] = *ei;
        acc_off += *ei;
    }

    auto sel = [&](const Def* cond, const Def* t, const Def* f) { return w.extract(w.tuple({f, t}), cond); };

    auto compute = [&](const DefVec& iters, const Def* ins, const Def* mem) -> std::pair<const Def*, const Def*> {
        auto o_ax      = iters[axn];
        const Def* cur = mem;
        // Read input `i` at `iters`, but with the `ax` coordinate shifted by off#i and clamped into input `i`.
        auto read_i = [&](u64 i) -> const Def* {
            auto in_buf          = ins->proj(nisn, i);
            auto [ibr, ibs, ibT] = Axm::isa<buffer::Buf>(in_buf->type())->args<3>();
            auto Sis_i           = Sis->proj(nisn, i);
            auto e_i_m1          = w.lit_i64(ext[i] - 1);
            auto loc             = w.call(core::wrap::sub, core::Mode::none, Defs{o_ax, off[i]});
            auto clamp           = w.call(core::extrema::smax,
                                          w.tuple({w.lit_i64(0), w.call(core::extrema::smin, w.tuple({loc, e_i_m1}))}));
            DefVec coords(rn);
            for (u64 d = 0; d < rn; ++d) {
                auto idx_i64 = (d == axn) ? clamp : iters[d];
                coords[d]    = w.call(core::conv::u, Sis_i->proj(rn, d), idx_i64);
            }
            auto coords_folded    = *Shape(w, coords).fold(Sis_i);
            auto [rd_mem, rd_val] = buffer::op_read(ibr, ibs, ibT, cur, in_buf, coords_folded)->projs<2>();
            cur                   = rd_mem;
            return rd_val;
        };
        // Select chain: the highest `i` with off#i ≤ o_ax owns the cell (offsets increase, later wins).
        auto result = read_i(0);
        for (u64 i = 1; i < nisn; ++i) {
            auto cond = w.call(core::icmp::uge, w.tuple({o_ax, off[i]}));
            result    = sel(cond, read_i(i), result);
        }
        return {cur, result};
    };

    return build_pointwise(w, result_ty, op_mem, op_is, s_out, rn, "concat", compute);
}

const Def* LowerMapReduce::lower_gather(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim]    = c->uncurry_args<3>();
    auto [T, r]               = Tr->projs<2>();
    auto [s_src, s_idx]       = shapes->projs<2>();
    auto [op_mem, input, idx] = rewrite(app->arg())->projs<3>();
    auto result_ty            = rewrite(app->type());

    auto r_l   = Lit::isa<u64>(r);
    auto dim_l = Lit::isa<u64>(dim);
    if (!r_l || !dim_l) {
        log().w("{} doesn't have lowering-time known rank/axis", app);
        return RWPhase::rewrite_imm_App(app);
    }
    auto rn = *r_l, axis = *dim_l;

    auto compute = [&](Defs iters, const Def* ins, const Def* mem) -> std::pair<const Def*, const Def*> {
        auto [in_buf, idx_buf] = ins->projs<2>();
        auto [ibr, ibs, ibT]   = Axm::isa<buffer::Buf>(in_buf->type())->args<3>();
        auto [xbr, xbs, xbT]   = Axm::isa<buffer::Buf>(idx_buf->type())->args<3>();

        DefVec idx_coords(rn);
        for (u64 d = 0; d < rn; ++d)
            idx_coords[d] = w.call(core::conv::u, s_idx->proj(rn, d), iters[d]);
        auto idx_folded          = *Shape(w, idx_coords).fold(s_idx);
        auto [idx_mem, selected] = buffer::op_read(xbr, xbs, xbT, mem, idx_buf, idx_folded)->projs<2>();
        auto selected_i64        = w.call<core::bitcast>(w.type_i64(), selected);

        DefVec src_coords(rn);
        for (u64 d = 0; d < rn; ++d) {
            auto coordinate = d == axis ? selected_i64 : iters[d];
            src_coords[d]   = w.call(core::conv::u, s_src->proj(rn, d), coordinate);
        }

        auto folded          = *Shape(w, src_coords).fold(s_src);
        auto [read_mem, val] = buffer::op_read(ibr, ibs, ibT, idx_mem, in_buf, folded)->projs<2>();

        return {read_mem, val};
    };
    return build_pointwise(w, result_ty, op_mem, w.tuple({input, idx}), s_idx, rn, "gather", compute);
}

const Def* LowerMapReduce::lower_scatter(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim]             = c->uncurry_args<3>();
    auto [T, r]                        = Tr->projs<2>();
    auto [s_src, s_idx, s_updates]     = shapes->projs<3>();
    auto [op_mem, input, idx, updates] = rewrite(app->arg())->projs<4>();
    auto result_ty                     = rewrite(app->type());

    auto r_l   = Lit::isa<u64>(r);
    auto dim_l = Lit::isa<u64>(dim);
    if (!r_l || !dim_l) {
        log().w("{} doesn't have lowering-time known rank/axis", app);
        return RWPhase::rewrite_imm_App(app);
    }
    auto rn = *r_l, axis = *dim_l;

    auto mem_ty = w.call<mem::M>(0);
    auto fun    = w.mut_fun(w.sigma({mem_ty, input->type(), idx->type(), updates->type()}), result_ty)->set("scatter");
    auto call   = w.app(cps::op_cps2ds_dep(fun), w.tuple({op_mem, input, idx, updates}));
    auto [fun_mem, in_buf, idx_buf, update_buf] = fun->var(2, 0)->projs<4>();
    auto cont                                   = fun->var(2, 1);

    auto [obr, obs, obT]  = Axm::isa<buffer::Buf>(result_ty->proj(2, 1))->args<3>();
    auto [a_mem, out_buf] = buffer::op_alloc(obr, obs, obT, fun_mem)->projs<2>();
    auto copy_mem         = buffer::op_copy(obr, obs, obT, a_mem, out_buf, in_buf);
    const Def* acc        = w.tuple({copy_mem, out_buf});
    auto current          = fun;
    DefVec iters;
    iters.reserve(rn);
    for (u64 d = 0; d < rn; ++d) {
        auto bound                  = w.call<core::bitcast>(w.type_i64(), s_idx->proj(rn, d));
        auto [body, for_call]       = counting_for(bound, acc, cont, w.sym("scatter_" + std::to_string(d)));
        auto [iter, new_acc, yield] = body->vars<3>();
        cont                        = yield;
        acc                         = new_acc;
        iters.push_back(iter);
        current->set(true, for_call);
        current = body;
    }
    auto [loop_mem, loop_buf] = acc->projs<2>();
    auto [xbr, xbs, xbT]      = Axm::isa<buffer::Buf>(idx_buf->type())->args<3>();
    auto [ubr, ubs, ubT]      = Axm::isa<buffer::Buf>(update_buf->type())->args<3>();

    DefVec idx_coords(rn);
    for (u64 d = 0; d < rn; ++d)
        idx_coords[d] = w.call(core::conv::u, s_idx->proj(rn, d), iters[d]);
    auto folded_idx = *Shape(w, idx_coords).fold(s_idx);
    DefVec update_coords(rn);
    for (u64 d = 0; d < rn; ++d)
        update_coords[d] = w.call(core::conv::u, s_updates->proj(rn, d), iters[d]);
    auto update_folded        = *Shape(w, update_coords).fold(s_updates);
    auto [idx_mem, selected]  = buffer::op_read(xbr, xbs, xbT, loop_mem, idx_buf, folded_idx)->projs<2>();
    auto [update_mem, update] = buffer::op_read(ubr, ubs, ubT, idx_mem, update_buf, update_folded)->projs<2>();
    auto selected_i64         = w.call<core::bitcast>(w.type_i64(), selected);

    DefVec dst_coords(rn);
    for (u64 d = 0; d < rn; ++d) {
        auto coordinate = d == axis ? selected_i64 : iters[d];
        dst_coords[d]   = w.call(core::conv::u, s_src->proj(rn, d), coordinate);
    }
    auto dst_folded           = *Shape(w, dst_coords).fold(s_src);
    auto [write_mem, written] = buffer::op_write(obr, obs, obT, update_mem, loop_buf, dst_folded, update)->projs<2>();
    current->app(true, cont, w.tuple({write_mem, loop_buf}));
    return call;
}

} // namespace mim::plug::btensor::phase
