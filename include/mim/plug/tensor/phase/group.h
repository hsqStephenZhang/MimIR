#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class Group : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    void start() override;
    const Def* rewrite_imm_App(const App*) final;

    /// The old `map_reduce_post` apps that root a group, with the dim of the loop their producers join.
    DefMap<u64> roots_;
};

} // namespace mim::plug::tensor::phase
