#include "mim/plug/tensor/phase/conv_to_img2col.h"

#include <algorithm>

#include "mim/def.h"
#include "mim/lam.h"

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

const Def* ConvToImg2Col::rewrite_via_impl(const App* app, const Def* impl_annex) {
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

const Def* ConvToImg2Col::rewrite_imm_App(const App* app) {
    if (!Axm::isa<tensor::conv>(app)) return RWPhase::rewrite_imm_App(app);

    // Today the Mim template supports the same NCHW/OIHW, groups=1, no-bias contract as `%tensor.conv`.
    // Future legality checks can live here, e.g. choosing direct lowering for tiny kernels or rejecting layouts.
    return rewrite_via_impl(app, new_world().annex<tensor::conv_img2col_impl>());
}

} // namespace mim::plug::tensor::phase
