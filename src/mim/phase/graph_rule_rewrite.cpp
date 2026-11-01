#include "mim/phase/graph_rule_rewrite.h"

#include <array>
#include <set>
#include <stdexcept>
#include <utility>

#include "mim/pattern/graph_rule.h"

namespace mim {

namespace {
std::string_view search_status(pattern::SearchStatus status) {
    switch (status) {
        case pattern::SearchStatus::Complete: return "Complete";
        case pattern::SearchStatus::Unknown: return "Unknown";
        case pattern::SearchStatus::Budget_Exceeded: return "Budget_Exceeded";
        case pattern::SearchStatus::Result_Limit: return "Result_Limit";
    }
    std::unreachable();
}
} // namespace

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
            std::optional<pattern::GraphRewrite> accepted;
            std::string reason = "no accepted match visited";
            pattern::GraphMatchOptions options;
            options.max_results = 1;
            options.accept      = [&](const pattern::GraphMatch& match) {
                auto candidate = compiled.plan(match, view);
                if (!candidate.plan) {
                    reason = candidate.reason;
                    if (candidate.status == pattern::MatchStatus::Unknown) {
                        profile_count("graph_rule.unknown");
                        log().d("graph_rule_app skipped a match in {}: {}", fn, candidate.reason);
                    }
                    return false;
                }
                std::array plans{candidate.plan.get()};
                auto next = pattern::rewrite_graph(view, plans);
                if (!next.is_changed()) {
                    profile_count("graph_rule.noop");
                    reason = "canonical no-op";
                    return false;
                }
                accepted.emplace(std::move(next));
                reason = "first accepted changed rewrite";
                return true;
            };
            auto matches = compiled.find(view, options);
            log().d("graph_rule_app search in {}: {} ({})", fn, search_status(matches.status), reason);
            if (accepted) {
                accepted->publish(view);
                invalidate();
                profile_count("graph_rule.rewrites");
                break; // The next invocation rebuilds matches and processes further groups.
            }
            if (matches.status != pattern::SearchStatus::Complete) profile_count("graph_rule.incomplete");
        }
    }
}

} // namespace mim
