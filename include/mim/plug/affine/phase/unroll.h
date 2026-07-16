#pragma once

#include <mim/phase.h>

namespace mim::plug::affine::phase {

/// Expands affine.unroll with literal bounds into straight-line CPS.
class Unroll : public RWPhase {
public:
    Unroll(World& world, flags_t annex)
        : RWPhase(world, annex) {}

    const Def* rewrite_imm_App(const App*) final;
};

} // namespace mim::plug::affine::phase
