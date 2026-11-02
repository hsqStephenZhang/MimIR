#pragma once

#include "mim/phase.h"

#include "mim/pattern/guard.h"
#include "mim/pattern/rewrite.h"

namespace mim {

/// Explicitly pure primitives and multi-root Rule templates survive intervening World rebuilds.
class GraphRuleRewrite : public Phase {
public:
    using Phase::Phase;
    GraphRuleRewrite(World&, std::vector<const Rule*>, std::vector<const Axm*> pure, bool skip_unproved = false);
    void apply(const App*) override;
    std::unique_ptr<Phase> recreate() override;
    void set_rewrite_options(pattern::RewriteOptions options) { rewrite_options_ = options; }

private:
    void start() override;
    std::shared_ptr<World> templates_;
    std::vector<const Rule*> rules_;
    std::vector<const Axm*> pure_;
    std::vector<pattern::DataOp> schemas_;
    pattern::RewriteOptions rewrite_options_;
    bool skip_unproved_ = false;
};

} // namespace mim
