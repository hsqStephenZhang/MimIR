#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Realizes `tensor.tile_req`: tiles the root at its parallel level and fuses the group's members into the tiles.
class TileFuse : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    void start() override;
    const Def* rewrite_imm_App(const App*) override;
    const Def* realize(const App* request, const DefSet& outside);

    DefMap<const Def*> replacements_;
};

} // namespace mim::plug::tensor::phase
