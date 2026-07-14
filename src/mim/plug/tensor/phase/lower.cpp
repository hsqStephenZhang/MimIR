#include "mim/plug/tensor/phase/lower.h"

#include "mim/def.h"
#include "mim/lam.h"

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

const Def* Lower::lower_via_impl(const App* app, const Def* impl_annex) {
    auto& w = new_world();

    // Walk the curry chain (innermost App outermost in syntax) to collect the args
    // in the order they were applied.
    DefVec args;
    const Def* head = app;
    while (auto h = head->isa<App>()) {
        args.push_back(rewrite(h->arg()));
        head = h->callee();
    }
    std::reverse(args.begin(), args.end());

    auto impl = impl_annex;
    for (auto a : args)
        impl = w.app(impl, a);

    // The `_impl` is a `lam`, so applying it triggers beta-reduction. Each `_impl`
    // body references the `_impl` variants of its dependencies directly, so the
    // chain bottoms out at the low-level axioms (`map_reduce`, …) in one go.
    return impl;
}

const Def* Lower::rewrite_imm_App(const App* app) {
    auto via_impl = [&]<class Src, class Impl>() -> const Def* {
        static_assert(Annex::base<Src>() != flags_t(-1), "invalid source axiom");
        static_assert(Annex::base<Impl>() != flags_t(-1), "invalid implementation axiom");
        if (!Axm::isa<Src>(app)) return nullptr;
        return lower_via_impl(app, new_world().annex<Impl>());
    };

    const Def* lowered = nullptr;
#define MIM_TRY_VIA_IMPL(src, impl) \
    if (!lowered) lowered = via_impl.template operator()<tensor::src, tensor::impl>()
    MIM_TRY_VIA_IMPL(broadcast_in_dim, broadcast_in_dim_impl);
    MIM_TRY_VIA_IMPL(product_2d, product_2d_impl);
    MIM_TRY_VIA_IMPL(bmm, bmm_impl);
    MIM_TRY_VIA_IMPL(dot_product, dot_product_impl);
    MIM_TRY_VIA_IMPL(transpose, transpose_impl);
    MIM_TRY_VIA_IMPL(transpose_2d, transpose_2d_impl);
    MIM_TRY_VIA_IMPL(map, map_impl);
    MIM_TRY_VIA_IMPL(unary, unary_impl);
    MIM_TRY_VIA_IMPL(relu, relu_impl);
    MIM_TRY_VIA_IMPL(binary, binary_impl);
    MIM_TRY_VIA_IMPL(select, select_impl);
    MIM_TRY_VIA_IMPL(repeat, repeat_impl);
    MIM_TRY_VIA_IMPL(reshape, reshape_impl);
    MIM_TRY_VIA_IMPL(slice, slice_impl);
    MIM_TRY_VIA_IMPL(flip, flip_impl);
    MIM_TRY_VIA_IMPL(conv, conv_impl);
    MIM_TRY_VIA_IMPL(pool, pool_impl);
    MIM_TRY_VIA_IMPL(gather, gather_impl);
    MIM_TRY_VIA_IMPL(scatter, scatter_impl);
#undef MIM_TRY_VIA_IMPL

    if (lowered) return lowered;

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::tensor::phase
