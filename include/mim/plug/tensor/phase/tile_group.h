#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Partitions the program into tile-and-fuse groups and wraps each root into a `tensor.tile_req`.
/// Roots are contractions, then other reductions, then the elementwise ops no group took;
/// members are the elementwise producers a root reaches through its inputs.
class TileGroup : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    struct Group {
        u64 kind;
        DefVec members;
    };

    void start() override;
    const Def* rewrite_imm_App(const App*) override;

    DefMap<Group> groups_;
};

} // namespace mim::plug::tensor::phase
