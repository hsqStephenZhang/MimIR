#include "mim/plug/xform/phase/schedule_for.h"

#include <algorithm>
#include <vector>

#include "mim/def.h"
#include "mim/lattice.h"
#include "mim/lam.h"
#include "mim/tuple.h"

#include "mim/plug/affine/affine.h"
#include "mim/plug/core/core.h"
#include "mim/plug/xform/xform.h"

namespace mim::plug::xform::phase {

namespace {

bool depends_on(const Def* def, const Def* var) {
    auto v = var ? var->isa<Var>() : nullptr;
    return v && def->free_vars().contains(v);
}

bool any_depends_on(std::initializer_list<const Def*> defs, const Def* var) {
    return std::ranges::any_of(defs, [var](const Def* def) { return depends_on(def, var); });
}

u64 ceil_div(u64 x, u64 y) { return y == 0 ? 0 : (x + y - 1) / y; }

struct LoopFrame {
    Lam* body_lam;
    const Def* exit;
    const Def* begin;
    const Def* end;
    const Def* step;
    const Def* init;

    const Def* iter() const { return body_lam->var(0); }
    const Def* acc() const { return body_lam->var(1); }
    const Def* yield() const { return body_lam->var(2); }
};

} // namespace

const Def* ScheduleFor::rewrite_via_impl(const App* app, const Def* impl_annex) {
    auto& w = new_world();

    DefVec args;
    const Def* head = app;
    while (auto h = head->isa<App>()) {
        args.push_back(rewrite(h->arg()));
        head = h->callee();
    }
    std::reverse(args.begin(), args.end());

    auto impl = impl_annex;
    for (auto arg : args)
        impl = w.app(impl, arg);

    return impl;
}

const Def* ScheduleFor::rewrite_via_impl(const App* app, const Def* impl_annex, const Def* extra_arg) {
    auto impl = rewrite_via_impl(app, impl_annex);
    return new_world().app(impl, rewrite(extra_arg));
}

std::optional<u64> ScheduleFor::static_u64(const Def* def) {
    return Lit::isa<u64>(rewrite(def));
}

const Def* ScheduleFor::rewrite_split_for_1d(const App* app) {
    auto split = Axm::isa<xform::split_for_1d>(app);
    auto [factor, old_body, old_exit, old_args] = split->uncurry_args<4>();
    auto [begin, end, step, old_init]           = old_args->projs<4>();
    (void)old_body;
    (void)old_exit;
    (void)old_init;

    auto factor_lit = static_u64(factor);
    auto begin_lit  = static_u64(begin);
    auto end_lit    = static_u64(end);
    auto step_lit   = static_u64(step);

    if (!factor_lit || !begin_lit || !end_lit || !step_lit || *factor_lit == 0 || *step_lit == 0)
        return rewrite_via_impl(app, new_world().annex<xform::split_for_1d_impl>());

    const auto begin_v = *begin_lit;
    const auto end_v   = *end_lit;
    const auto step_v  = *step_lit;
    const auto factor_v = *factor_lit;
    const auto trip    = end_v <= begin_v ? 0 : (end_v - begin_v + step_v - 1) / step_v;

    if (trip % factor_v == 0) return rewrite_via_impl(app, new_world().annex<xform::split_for_1d_no_tail_impl>());

    const auto full_trip = (trip / factor_v) * factor_v;
    const auto main_end  = begin_v + full_trip * step_v;
    return rewrite_via_impl(app, new_world().annex<xform::split_for_1d_tail_impl>(), begin->world().lit(begin->type(), main_end));
}

const Def* ScheduleFor::rewrite_split(const App* app) {
    auto& w = new_world();
    auto split = Axm::isa<xform::split>(app);
    auto [factors, old_body, old_exit, old_args] = split->uncurry_args<4>();
    auto [begin, end, step, old_init] = old_args->projs<4>();

    auto fallback = [&]() {
        return w.call<affine::For>(rewrite(old_body), rewrite(old_exit), rewrite(old_args));
    };

    auto factors_tuple = rewrite(factors)->isa<Tuple>();
    auto body_lam = old_body->isa_mut<Lam>();
    auto idx_size = Idx::isa(begin->type());
    if (!factors_tuple || factors_tuple->num_ops() == 0 || !body_lam || !idx_size) return fallback();
    auto idx_size_new = rewrite(idx_size);
    auto begin_new = rewrite(begin);
    auto end_new = rewrite(end);
    auto step_new = rewrite(step);
    auto idx_type = begin_new->type();

    std::vector<const Def*> split_factors;
    split_factors.reserve(factors_tuple->num_ops());

    std::optional<size_t> infer_pos;
    std::optional<u64> known_product = 1;
    for (size_t i = 0; i < factors_tuple->num_ops(); ++i) {
        auto factor = factors_tuple->op(i);
        if (factor->isa<Top>() && factor->type() == w.type_nat()) {
            if (infer_pos) return fallback();
            infer_pos = i;
            split_factors.push_back(nullptr);
            continue;
        }

        if (auto lit = Lit::isa<u64>(factor)) {
            if (*lit == 0) return fallback();
            if (known_product) *known_product *= *lit;
        } else {
            known_product = {};
        }
        split_factors.push_back(factor);
    }

    auto begin_lit = static_u64(begin);
    auto end_lit = static_u64(end);
    auto step_lit = static_u64(step);
    std::optional<u64> trip;
    if (begin_lit && end_lit && step_lit && *step_lit != 0)
        trip = *end_lit <= *begin_lit ? 0 : ceil_div(*end_lit - *begin_lit, *step_lit);

    if (infer_pos) {
        if (!trip || !known_product || *known_product == 0) return fallback();
        auto inferred = *trip == 0 ? 0 : ceil_div(*trip, *known_product);
        split_factors[*infer_pos] = w.lit_nat(inferred);
        *known_product *= inferred;
    } else if (trip && known_product && *known_product < *trip) {
        return fallback();
    }

    auto zero = w.lit(idx_type, 0);
    auto one = w.lit(idx_type, 1);

    DefVec factor_idxs;
    factor_idxs.reserve(split_factors.size());
    for (auto factor : split_factors)
        factor_idxs.push_back(w.call<core::idx>(idx_size_new, core::Mode::nuw, factor));

    std::vector<Lam*> loops;
    loops.reserve(split_factors.size());
    auto loop_dom = rewrite(body_lam->dom());
    for (size_t i = 0; i < split_factors.size(); ++i)
        loops.push_back(w.mut_con(loop_dom)->set("split_loop"));

    auto linear = loops.front()->var(0);
    for (size_t i = 1; i < loops.size(); ++i) {
        linear = w.call(core::wrap::mul, core::Mode::nuw, Defs{linear, factor_idxs[i]});
        linear = w.call(core::wrap::add, core::Mode::nuw, Defs{linear, loops[i]->var(0)});
    }
    auto offset = w.call(core::wrap::mul, core::Mode::nuw, Defs{linear, step_new});
    auto iter = w.call(core::wrap::add, core::Mode::nuw, Defs{begin_new, offset});
    auto in_bounds = w.call(core::icmp::ul, Defs{iter, end_new});

    auto skip = w.mut_con(loop_dom)->set("split_skip");
    skip->set(true, w.app(skip->var(2), skip->var(1)));
    auto branch = w.extract(w.tuple({skip, rewrite(body_lam)}), in_bounds);
    loops.back()->set(true, w.app(branch, Defs{iter, loops.back()->var(1), loops.back()->var(2)}));

    for (size_t pos = loops.size() - 1; pos-- > 0;) {
        auto args = w.tuple({zero, factor_idxs[pos + 1], one, loops[pos]->var(1)});
        loops[pos]->set(true, w.call<affine::For>(loops[pos + 1], loops[pos]->var(2), args));
    }

    auto root_args = w.tuple({zero, factor_idxs.front(), one, rewrite(old_init)});
    return w.call<affine::For>(loops.front(), rewrite(old_exit), root_args);
}

const Def* ScheduleFor::rewrite_reorder(const App* app) {
    auto& w = new_world();
    auto reorder = Axm::isa<xform::reorder>(app);
    auto [order, root_body, root_exit, root_args] = reorder->uncurry_args<4>();
    auto [root_begin, root_end, root_step, root_init] = root_args->projs<4>();

    auto fallback = [&]() {
        return w.call<affine::For>(rewrite(root_body), rewrite(root_exit), rewrite(root_args));
    };

    auto order_tuple = rewrite(order)->isa<Tuple>();
    if (!order_tuple || order_tuple->num_ops() == 0) return fallback();

    std::vector<size_t> perm;
    perm.reserve(order_tuple->num_ops());
    std::vector<bool> seen(order_tuple->num_ops(), false);
    for (auto op : order_tuple->ops()) {
        auto index = Lit::isa<u64>(op);
        if (!index || *index >= order_tuple->num_ops() || seen[*index]) return fallback();
        seen[*index] = true;
        perm.push_back(*index);
    }

    std::vector<LoopFrame> loops;
    loops.reserve(perm.size());

    auto body_lam = root_body->isa_mut<Lam>();
    if (!body_lam) return fallback();
    loops.push_back({body_lam, root_exit, root_begin, root_end, root_step, root_init});

    for (size_t i = 1; i < perm.size(); ++i) {
        auto inner_for = Axm::isa<affine::For>(loops.back().body_lam->body());
        if (!inner_for) return fallback();

        auto [inner_body, inner_exit, inner_args] = inner_for->uncurry_args<3>();
        auto [inner_begin, inner_end, inner_step, inner_init] = inner_args->projs<4>();
        auto inner_lam = inner_body->isa_mut<Lam>();
        if (!inner_lam) return fallback();

        // Strict perfect nest: each inner loop threads exactly the parent accumulator and continuation.
        if (inner_exit != loops.back().yield()) return fallback();
        if (inner_init != loops.back().acc()) return fallback();

        loops.push_back({inner_lam, inner_exit, inner_begin, inner_end, inner_step, inner_init});
    }

    auto leaf = loops.back().body_lam;

    // First implementation is intentionally rectangular. This enforces TVM's "outer domain cannot depend on inner
    // loops" condition by rejecting all loop-carried bound dependencies.
    for (auto& loop : loops) {
        for (auto& other : loops) {
            if (any_depends_on({loop.begin, loop.end, loop.step, loop.body_lam->filter()}, other.iter())) return fallback();
        }
    }

    // Keep continuation threading simple: the leaf may use all loop iterators, plus only the innermost acc/yield pair.
    for (size_t i = 0; i + 1 < loops.size(); ++i) {
        if (any_depends_on({leaf->filter(), leaf->body()}, loops[i].acc())) return fallback();
        if (any_depends_on({leaf->filter(), leaf->body()}, loops[i].yield())) return fallback();
    }

    std::vector<Lam*> new_lams;
    new_lams.reserve(perm.size());
    for (auto old_index : perm) {
        auto lam = w.mut_con(rewrite(loops[old_index].body_lam->dom()))->set("reorder_loop");
        new_lams.push_back(lam);
    }

    auto new_lam_for_old = [&](size_t old_index) -> Lam* {
        auto it = std::ranges::find(perm, old_index);
        return new_lams[std::distance(perm.begin(), it)];
    };

    push();
    for (size_t old_index = 0; old_index < loops.size(); ++old_index)
        map(loops[old_index].iter(), new_lam_for_old(old_index)->var(0));
    map(leaf->var(1), new_lams.back()->var(1));
    map(leaf->var(2), new_lams.back()->var(2));
    new_lams.back()->set(rewrite(leaf->filter()), rewrite(leaf->body()));
    pop();

    for (size_t pos = new_lams.size() - 1; pos-- > 0;) {
        auto old_index = perm[pos + 1];
        auto args = w.tuple({
            rewrite(loops[old_index].begin),
            rewrite(loops[old_index].end),
            rewrite(loops[old_index].step),
            new_lams[pos]->var(1),
        });
        new_lams[pos]->set(rewrite(loops[perm[pos]].body_lam->filter()),
                           w.call<affine::For>(new_lams[pos + 1], new_lams[pos]->var(2), args));
    }

    auto outer_old_index = perm.front();
    auto outer_args = w.tuple({
        rewrite(loops[outer_old_index].begin),
        rewrite(loops[outer_old_index].end),
        rewrite(loops[outer_old_index].step),
        rewrite(root_init),
    });
    return w.call<affine::For>(new_lams.front(), rewrite(root_exit), outer_args);
}

const Def* ScheduleFor::rewrite_imm_App(const App* app) {
    if (Axm::isa<xform::split>(app)) return rewrite_split(app);
    if (Axm::isa<xform::split_for_1d>(app)) return rewrite_split_for_1d(app);
    if (Axm::isa<xform::reorder>(app)) return rewrite_reorder(app);

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::xform::phase
