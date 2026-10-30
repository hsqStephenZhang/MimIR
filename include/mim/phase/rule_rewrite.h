#pragma once

#include "mim/phase.h"

#include "mim/pattern/rule.h"

namespace mim {

/// One bottom-up graph rebuild with an explicitly ordered list of closed Rule templates.
/// The first accepted rule wins; new RHS nodes are considered on the next PhaseMan round.
class RuleRewrite : public RWPhase {
public:
    explicit RuleRewrite(World& world)
        : RWPhase(world, "rule_app") {}
    RuleRewrite(World& world, flags_t annex)
        : RWPhase(world, annex) {}
    RuleRewrite(World&, std::vector<const Rule*>);

    void apply(const App*) override;
    std::unique_ptr<Phase> recreate() override;

private:
    void set_rules(std::vector<const Rule*>);
    void start() override;
    const Def* rewrite_imm(const Def*) override;
    const Def* rewrite_imm_Rule(const Rule*) override;
    const Def* rewrite_mut_Rule(Rule*) override;

    // Other phases may destroy every Def in the program World. Keep templates in their own World,
    // and import fresh matching instances at the beginning of each run.
    std::shared_ptr<World> template_world_;
    std::vector<const Rule*> templates_;
    std::vector<std::unique_ptr<pattern::RulePattern>> patterns_;
};

} // namespace mim
