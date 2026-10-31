#pragma once

#include "mim/phase.h"

namespace mim {

/// Explicitly pure primitives and multi-root Rule templates survive intervening World rebuilds.
class GraphRuleRewrite : public Phase {
public:
    using Phase::Phase;
    void apply(const App*) override;
    std::unique_ptr<Phase> recreate() override;

private:
    void start() override;
    std::shared_ptr<World> templates_;
    std::vector<const Rule*> rules_;
    std::vector<const Axm*> pure_;
};

} // namespace mim
