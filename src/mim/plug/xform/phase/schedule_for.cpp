#include "mim/plug/xform/phase/schedule_for.h"

#include <algorithm>

#include "mim/def.h"
#include "mim/lam.h"
#include "mim/tuple.h"

#include "mim/plug/affine/affine.h"
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

const Def* ScheduleFor::rewrite_exchange_for_2d(const App* app) {
    auto& w = new_world();
    auto exchange = Axm::isa<xform::exchange_for_2d>(app);
    auto [old_outer_body, old_exit, old_outer_args]  = exchange->uncurry_args<3>();
    auto [old_obegin, old_oend, old_ostep, old_init] = old_outer_args->projs<4>();

    auto fallback = [&]() {
        return w.call<affine::For>(rewrite(old_outer_body), rewrite(old_exit), rewrite(old_outer_args));
    };

    auto old_outer_lam = old_outer_body->isa_mut<Lam>();
    if (!old_outer_lam) return fallback();

    auto old_inner_for = Axm::isa<affine::For>(old_outer_lam->body());
    if (!old_inner_for) return fallback();

    auto [old_inner_body, old_inner_exit, old_inner_args] = old_inner_for->uncurry_args<3>();
    auto [old_ibegin, old_iend, old_istep, old_iinit]     = old_inner_args->projs<4>();
    auto old_inner_lam                                    = old_inner_body->isa_mut<Lam>();
    if (!old_inner_lam) return fallback();

    auto old_outer_iter  = old_outer_lam->var(0);
    auto old_outer_acc   = old_outer_lam->var(1);
    auto old_outer_yield = old_outer_lam->var(2);
    auto old_inner_iter  = old_inner_lam->var(0);
    auto old_inner_acc   = old_inner_lam->var(1);
    auto old_inner_yield = old_inner_lam->var(2);

    // Conservative perfect-nest check: the inner loop must thread exactly the outer accumulator and yield exactly to
    // the outer continuation. Bounds must describe a rectangular iteration domain.
    if (old_inner_exit != old_outer_yield) return fallback();
    if (old_iinit != old_outer_acc) return fallback();
    if (depends_on(old_outer_lam->filter(), old_outer_iter)) return fallback();
    if (any_depends_on({old_ibegin, old_iend, old_istep}, old_outer_iter)) return fallback();
    if (any_depends_on({old_inner_lam->filter(), old_inner_lam->body()}, old_outer_yield)) return fallback();

    auto new_outer = w.mut_con(rewrite(old_inner_lam->dom()))->set("exchange_outer");
    auto new_inner = w.mut_con(rewrite(old_outer_lam->dom()))->set("exchange_inner");

    // Rebuild the original inner body under swapped loop variables:
    //   old outer iter -> new inner iter
    //   old inner iter -> new outer iter
    push();
    map(old_outer_iter, new_inner->var(0));
    map(old_outer_acc, new_inner->var(1));
    map(old_outer_yield, new_outer->var(2));
    map(old_inner_iter, new_outer->var(0));
    map(old_inner_acc, new_inner->var(1));
    map(old_inner_yield, new_inner->var(2));
    new_inner->set(rewrite(old_inner_lam->filter()), rewrite(old_inner_lam->body()));
    pop();

    auto new_inner_args = w.tuple({rewrite(old_obegin), rewrite(old_oend), rewrite(old_ostep), new_outer->var(1)});
    new_outer->set(rewrite(old_outer_lam->filter()), w.call<affine::For>(new_inner, new_outer->var(2), new_inner_args));

    auto new_outer_args = w.tuple({rewrite(old_ibegin), rewrite(old_iend), rewrite(old_istep), rewrite(old_init)});
    return w.call<affine::For>(new_outer, rewrite(old_exit), new_outer_args);
}

const Def* ScheduleFor::rewrite_imm_App(const App* app) {
    if (Axm::isa<xform::split_for_1d>(app)) return rewrite_split_for_1d(app);
    if (Axm::isa<xform::exchange_for_2d>(app)) return rewrite_exchange_for_2d(app);

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::xform::phase
