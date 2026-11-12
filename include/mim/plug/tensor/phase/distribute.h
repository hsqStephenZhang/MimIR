#pragma once

#include <mim/phase.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

/// Keeps each lane's component of the warp values that `gpu.lanes`, `gpu.warp_mma` and `tensor.warp_keep` build.
/// A pack is reduced as soon as it is read, so the reads left over are those of these axioms; a read of the own
/// lane `self` becomes the local value, of another lane a shuffle of it.
class Distribute : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    void start() final;
    const Def* rewrite_imm_Extract(const Extract*) final;
    const Def* local(const App* warp);
    const Def* shuffle(const Def* local, const Def* lane);

    const Def* self_ = nullptr;
};

} // namespace mim::plug::tensor::phase
