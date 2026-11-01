#pragma once

#include "mim/pattern/rewrite.h"
#include "mim/pattern/rule.h"

namespace mim::pattern {

struct GraphRulePlan {
    MatchStatus status;
    std::unique_ptr<RewritePlan> plan;
    std::string reason;
};

class GraphRulePattern {
public:
    explicit GraphRulePattern(const Rule*, size_t max_roots = 64);

    GraphMatches find(const GraphView&, GraphMatchOptions = {}) const;
    GraphRulePlan plan(const GraphMatch&, const GraphView&) const;

private:
    RulePattern lowered_;
    GraphPattern graph_;
    std::vector<const Def*> lhs_, rhs_;
    std::vector<Anchor> anchors_;
    bool aggregate_rhs_ = false;
};

} // namespace mim::pattern
