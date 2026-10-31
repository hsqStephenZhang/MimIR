#include "mim/phase/graph_rule_rewrite.h"

#include <array>
#include <set>
#include <stdexcept>

#include "mim/pattern/graph_rule.h"

namespace mim {

void GraphRuleRewrite::apply(const App* app) {
    if (!app) throw std::invalid_argument("graph_rule_app requires (Rules, pure axioms)");
    auto pair = app->arg()->isa<Tuple>();
    if (!pair || pair->num_ops() != 2) app->blame("graph_rule_app requires a static (Rules, pure axioms) pair").bail();
    rules_.clear();
    pure_.clear();
    templates_ = world().inherit();
    Rewriter importer(*templates_);
    std::function<void(const Def*, bool)> collect = [&](const Def* def, bool rules) {
        if (auto tuple = def->isa<Tuple>()) {
            for (auto op : tuple->ops())
                collect(op, rules);
        } else if (auto pack = def->isa_imm<Pack>()) {
            auto count = Lit::isa(*pack->shape());
            if (!count) def->blame("graph_rule_app requires static lists").bail();
            if (*count) collect(pack->body(), rules);
        } else if (rules) {
            auto rule = def->isa<Rule>();
            if (!rule || !rule->is_set() || !rule->is_closed())
                def->blame("graph_rule_app expects closed Rules").bail();
            rules_.push_back(importer.rewrite(rule)->as<Rule>());
        } else {
            auto axm = def->isa<Axm>();
            if (!axm) def->blame("graph_rule_app purity declarations must be axioms").bail();
            pure_.push_back(importer.rewrite(axm)->as<Axm>());
        }
    };
    collect(pair->op(0), true);
    collect(pair->op(1), false);
}

std::unique_ptr<Phase> GraphRuleRewrite::recreate() {
    auto result        = std::make_unique<GraphRuleRewrite>(world(), annex());
    result->templates_ = templates_;
    result->rules_     = rules_;
    result->pure_      = pure_;
    return result;
}

void GraphRuleRewrite::start() {
    if (rules_.empty()) return;
    std::vector<Sym> names;
    for (auto [name, def] : world().externals())
        if (def->isa<Lam>()) names.push_back(name);
    for (auto name : names) {
        Rewriter importer(world());
        pattern::OpRegistry ops;
        std::set<flags_t> registered;
        for (auto source : pure_) {
            auto axm = importer.rewrite(source)->as<Axm>();
            if (!registered.insert(axm->flags()).second) continue;
            size_t groups = 0;
            for (auto type = axm->type(); auto pi = type->isa<Pi>(); type = pi->codom())
                if (++groups > 64) throw std::invalid_argument("graph_rule_app axiom curry depth exceeds 64");
            // Purity is a caller assertion. No data schema is inferred from raw operands, so
            // exclusive-use queries stay Unknown rather than treating metadata as tensor inputs.
            ops.add(
                axm->flags(),
                [groups](const Def* def, size_t) -> std::optional<pattern::OpView> {
                    size_t count = 0;
                    for (auto cur = def; auto app = cur->isa<App>(); cur = app->callee())
                        if (++count > groups) return {};
                    if (count != groups) return {};
                    return pattern::OpView{.inputs = {}, .complete = false};
                },
                true);
        }
        auto fn = world().externals()[name]->isa_mut<Lam>();
        if (!fn || !fn->is_set() || !fn->is_closed()) continue;
        for (auto source : rules_) {
            pattern::GraphRulePattern compiled(importer.rewrite(source)->as<Rule>());
            pattern::GraphView view({fn->body()}, {.operators = &ops, .scope = fn});
            auto matches = compiled.find(view);
            if (matches.status != pattern::SearchStatus::Complete) {
                profile_count("graph_rule.incomplete");
                continue;
            }
            bool changed = false;
            for (const auto& match : matches.matches) {
                auto candidate = compiled.plan(match, view);
                if (!candidate.plan) {
                    if (candidate.status == pattern::MatchStatus::Unknown) {
                        profile_count("graph_rule.unknown");
                        log().d("graph_rule_app skipped a match in {}: {}", fn, candidate.reason);
                    }
                    continue;
                }
                std::array plans{candidate.plan.get()};
                auto next = pattern::rewrite_graph(view, plans);
                next.publish(view);
                invalidate();
                profile_count("graph_rule.rewrites");
                changed = true;
                break;
            }
            if (changed) break; // The next invocation rebuilds matches and processes further groups.
        }
    }
}

} // namespace mim
