#pragma once

#include <mim/phase.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

/// Prints the synchronization events one thread runs in each ring, for the checkers in `proofs/ring`.
class RingTrace : public Phase {
public:
    RingTrace(World& world, flags_t annex)
        : Phase(world, annex) {}

private:
    void start() final;
};

} // namespace mim::plug::gpu::phase
