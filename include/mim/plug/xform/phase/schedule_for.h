#pragma once

#include <mim/phase.h>

namespace mim::plug::xform::phase {

/// Expands xform loop schedule markers into plain affine.For nests.
class ScheduleFor : public RWPhase {
public:
    ScheduleFor(World& world, flags_t annex)
        : RWPhase(world, annex) {}

private:
    const Def* rewrite_imm_App(const App*) final;

    const Def* rewrite_via_impl(const App*, const Def* impl_annex);
    const Def* rewrite_via_impl(const App*, const Def* impl_annex, const Def* extra_arg);
    std::optional<u64> static_u64(const Def*);
    const Def* rewrite_exchange_for_2d(const App*);
    const Def* rewrite_split_for_1d(const App*);
};

} // namespace mim::plug::xform::phase
