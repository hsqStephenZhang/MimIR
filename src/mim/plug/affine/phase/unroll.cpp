#include "mim/plug/affine/phase/unroll.h"

#include <mim/lam.h>

#include <algorithm>

#include "mim/plug/affine/affine.h"

namespace mim::plug::affine::phase {

namespace {

const Def* lower_via_impl(RWPhase& phase, const App* app, const Def* impl_annex) {
    auto& w = phase.new_world();

    DefVec args;
    const Def* head = app;
    while (auto h = head->isa<App>()) {
        args.push_back(phase.rewrite(h->arg()));
        head = h->callee();
    }
    std::reverse(args.begin(), args.end());

    auto impl = impl_annex;
    for (auto arg : args)
        impl = w.app(impl, arg);
    return impl;
}

} // namespace

const Def* Unroll::rewrite_imm_App(const App* app) {
    if (is_bootstrapping()) return RWPhase::rewrite_imm_App(app);

    if (auto unroll_ax = Axm::isa<affine::unroll>(app)) {
        DLOG("rewriting unroll axm: `{}`", unroll_ax);
        auto [body, exit, args]       = unroll_ax->uncurry_args<3>();
        auto [begin, end, step, init] = args->projs<4>([this](const Def* def) { return rewrite(def); });
        if (!Lit::isa<u64>(begin) || !Lit::isa<u64>(end) || !Lit::isa<u64>(step))
            error("affine.unroll requires literal begin/end/step, got ({}, {}, {})", begin, end, step);
        if (Lit::as<u64>(step) == 0) error("affine.unroll requires a non-zero step");
        return lower_via_impl(*this, app, new_world().annex<affine::unroll_impl>());
    }

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::affine::phase
