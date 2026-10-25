#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class Group : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    void start() override;
    const Def* rewrite_imm_App(const App*) final;
    /// The dim of the loop the producers of @p app join, unless its tiles would reach a root.
    std::optional<u64> grouped(const App* app);
    /// Whether tiles taking in @p p reach a root, through its producers too unless it groups them itself.
    bool reaches(const App* p);

    /// The old `map_reduce_post` apps that root a group, with the dim of the loop their producers join.
    DefMap<u64> roots_;
    DefSet taken_;
    DefMap<std::optional<u64>> dims_;
    DefMap<bool> reach_;
};

} // namespace mim::plug::tensor::phase
