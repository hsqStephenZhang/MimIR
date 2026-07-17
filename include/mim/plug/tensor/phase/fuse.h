#pragma once

#include <mim/phase.h>

#include "mim/plug/tensor/phase/fusion_partition.h"

namespace mim::plug::tensor::phase {

class Fuse : public RWPhase {
public:
    Fuse(World& world, flags_t annex)
        : RWPhase(world, annex, &partition_)
        , partition_(world) {}

private:
    const Def* rewrite_imm_App(const App*) final;

    const Def* fuse_map_reduce(const App*);

    FusionPartitionAnalysis partition_;
};

} // namespace mim::plug::tensor::phase
