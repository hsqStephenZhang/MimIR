#include "mim/plug/tensor/phase/lower.h"

#include <ranges>

#include <mim/def.h>
#include <mim/driver.h>
#include <mim/lam.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

const Def* Lower::fastest_axis_2(const App* app, const Def* rank) {
    auto& w = new_world();
    auto b  = rewrite(app->arg()->proj(2, 1));
    return w.app(w.app(w.annex<tensor::fastest_axis>(), b->type()), {rank, b});
}

const Def* Lower::lower_via_impl(const App* app, const Def* impl) {
    auto& w = new_world();

    // The curry chain, innermost App first — hence re-applied in reverse.
    auto args = DefVec();
    for (const App* h = app; h; h = h->callee()->isa<App>())
        args.emplace_back(rewrite(h->arg()));

    // The `_impl` is a `lam`, so applying it triggers beta-reduction. Each `_impl`
    // body references the `_impl` variants of its dependencies directly, so the
    // chain bottoms out at the low-level axioms (`map_reduce`, …) in one go.
    for (auto a : args | std::views::reverse)
        impl = w.app(impl, a);
    return impl;
}

const Def* Lower::rewrite_imm_App(const App* app) {
    if (Axm::isa<tensor::prepack>(app) && driver().arg_nat("tensor", "reference-tiles").value_or(0)) {
        auto& w                           = new_world();
        auto [meta, shape, layout, input] = app->uncurry_args<4>();
        auto stage = w.app(w.app(w.app(w.annex<tensor::stage>(), rewrite(meta)), rewrite(shape)), rewrite(layout));
        return w.app(w.app(stage, w.lit_nat_0()), rewrite(input));
    }
    if (Axm::isa<tensor::sdpa>(app)) return lower_via_impl(app, impl<tensor::attention_impl>());
    if (Axm::isa<tensor::sdpa_masked>(app)) return lower_via_impl(app, impl<tensor::attention_masked_impl>());
    if (Axm::isa<tensor::sdpa_online>(app)) return lower_via_impl(app, impl<tensor::sdpa_online_impl>());
    auto& w = new_world();

    if (Axm::isa<tensor::reverse_at>(app)) {
        auto [p, cs, dim, size] = app->args<4>();
        auto plan               = reverse_tiles(rewrite(p), rewrite(cs), rewrite(dim), rewrite(size));
        auto report             = w.log();
        report.set(fe::Log::Level::Warn)
            .w("tensor.reverse_at not realized: {}", plan.value ? "reverse_rule was not applied" : plan.reason);
        return w.tuple({rewrite(p), rewrite(cs)});
    }
    if (Axm::isa<tensor::tile_of>(app)) {
        auto [producer, off, ext] = app->args<3>();
        auto tiled                = producer_tile(rewrite(producer), rewrite(off), rewrite(ext));
        if (tiled.value) return tiled.value;
        auto report = w.log();
        report.set(fe::Log::Level::Warn).w("tensor.tile_of fallback: {}", tiled.reason);
        return lower_via_impl(app, impl<tensor::tile_of_ref>());
    }
    if (Axm::isa<tensor::tile_of_box>(app)) {
        auto [producer, off, ext, cap] = app->args<4>();
        auto tiled                     = producer_tile(rewrite(producer), rewrite(off), rewrite(ext), rewrite(cap));
        if (tiled.value) return tiled.value;
        auto report = w.log();
        report.set(fe::Log::Level::Warn).w("tensor.tile_of_box fallback: {}", tiled.reason);
        return lower_via_impl(app, impl<tensor::tile_of_box_ref>());
    }
    if (Axm::isa<tensor::tiled>(app) && driver().arg_nat("tensor", "reference-tiles").value_or(0))
        return lower_via_impl(app, impl<tensor::tiled_impl>());
    if (Axm::isa<tensor::tiled_multi>(app)) {
        grid_tile_plan(app);
        if (driver().arg_nat("tensor", "reference-tiles").value_or(0))
            return lower_via_impl(app, impl<tensor::tiled_multi_impl>());
    }
    if (Axm::isa<tensor::tiled_n>(app)) {
        multi_tile_plan(app);
        if (driver().arg_nat("tensor", "reference-tiles").value_or(0))
            return lower_via_impl(app, impl<tensor::tiled_n_impl>());
    }
    if (Axm::isa<tensor::tile_fold>(app)
        && (driver().arg_nat("tensor", "reference-tiles").value_or(0)
            || (!TensorType::isa(app->type()) && !app->type()->isa_imm<Sigma>())))
        return lower_via_impl(app, impl<tensor::tile_fold_impl>());
    if (Axm::isa<tensor::stage>(app) && driver().arg_nat("tensor", "reference-tiles").value_or(0))
        return lower_via_impl(app, impl<tensor::stage_impl>());
    if (Axm::isa<tensor::stage>(app)) {
        if (auto slice = Axm::isa<tensor::slice>(Axm::peel<tensor::buf>(app->arg())))
            return w.app(rewrite(app->callee()), RWPhase::rewrite_imm_App(slice));
    }
    if (Axm::isa<tensor::broadcast_in_dim>(app)) return lower_via_impl(app, impl<tensor::broadcast_in_dim_impl>());
    if (Axm::isa<tensor::transpose>(app)) return lower_via_impl(app, impl<tensor::transpose_impl>());
    if (Axm::isa<tensor::map>(app)) return lower_via_impl(app, impl<tensor::map_impl>());
    if (Axm::isa<tensor::unary>(app)) return lower_via_impl(app, impl<tensor::unary_impl>());
    if (Axm::isa<tensor::binary>(app)) return lower_via_impl(app, impl<tensor::binary_impl>());
    if (Axm::isa<tensor::select>(app)) return lower_via_impl(app, impl<tensor::select_impl>());
    if (Axm::isa<tensor::repeat>(app)) return lower_via_impl(app, impl<tensor::repeat_impl>());
    if (Axm::isa<tensor::reshape>(app)) {
        auto [meta, input_shape, output_shape, input] = app->uncurry_args<4>();
        if (*Shape(input_shape).fold() != *Shape(output_shape).fold())
            return lower_via_impl(app, impl<tensor::reshape_impl>());
    }
    if (Axm::isa<tensor::slice>(app)) return lower_via_impl(app, impl<tensor::slice_impl>());
    if (Axm::isa<tensor::flip>(app)) return lower_via_impl(app, impl<tensor::flip_impl>());
    if (Axm::isa<tensor::conv>(app)) return lower_via_impl(app, impl<tensor::conv_impl>());
    if (Axm::isa<tensor::conv_grouped>(app)) return lower_via_impl(app, impl<tensor::conv_grouped_impl>());
    if (Axm::isa<tensor::pool>(app)) return lower_via_impl(app, impl<tensor::pool_impl>());

    // The dot family's `_impl`s take a leading `fastest_2` with no axiom counterpart — the
    // `tensor.fastest_axis` reflection of the right operand, pre-applied here at the staging
    // point where that operand is concrete (see tensor.dot_product_impl for the decision).
    if (Axm::isa<tensor::product_2d>(app))
        return lower_via_impl(app, w.app(impl<tensor::product_2d_impl>(), fastest_axis_2(app, w.lit_nat(2))));
    if (Axm::isa<tensor::bmm>(app))
        return lower_via_impl(app, w.app(impl<tensor::bmm_impl>(), fastest_axis_2(app, w.lit_nat(3))));
    if (Axm::isa<tensor::dot_product>(app)) {
        // The curry chain, outermost app first: [a, b] {s1 s2} [c1, c2, b1, b2] {nc nb} {r1 r2};
        // the right operand's rank is {r1 r2}#1.
        auto groups = app->callee()->as<App>()->callee()->as<App>()->callee()->as<App>();
        auto r2     = groups->callee()->as<App>()->arg()->proj(2, 1);
        return lower_via_impl(app, w.app(impl<tensor::dot_product_impl>(), fastest_axis_2(app, rewrite(r2))));
    }

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::tensor::phase
