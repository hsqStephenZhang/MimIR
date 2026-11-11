#pragma once

#include <mim/phase.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

/// Checks the use discipline of work-group phase tokens and panels that their types cannot express.
class CheckSync : public Phase {
public:
    CheckSync(World& world, flags_t annex)
        : Phase(world, annex) {}

private:
    void start() final;
};

} // namespace mim::plug::gpu::phase
