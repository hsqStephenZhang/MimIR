#pragma once

#include <mim/phase.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

/// Lowers the work-group phase tokens and panels to plain memory, buffers and `gpu.fence`s.
class ErasePhases : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App*) final;
};

} // namespace mim::plug::gpu::phase
