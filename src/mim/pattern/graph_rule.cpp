#include "mim/pattern/graph_rule.h"

#include <functional>
#include <stdexcept>

#include "mim/rewrite.h"
#include "mim/world.h"

#include "mim/pattern/guard.h"

namespace mim::pattern {

namespace {

std::vector<const Def*> roots(const Def* value, size_t limit) {
    if (!limit) throw std::invalid_argument("graph Rule root budget is zero");
    if (auto tuple = value->isa<Tuple>()) {
        if (!tuple->num_ops() || tuple->num_ops() > limit)
            throw std::invalid_argument("graph Rule root budget exceeded");
        return {tuple->ops().begin(), tuple->ops().end()};
    }
    if (auto pack = value->isa_imm<Pack>()) {
        auto n = Lit::isa(*pack->shape());
        if (!n || !*n || *n > limit)
            throw std::invalid_argument("graph Rule needs a bounded, one-dimensional root list");
        return std::vector<const Def*>(*n, pack->body());
    }
    return {value};
}

} // namespace

GraphRulePattern::GraphRulePattern(const Rule* rule, size_t max_roots)
    : lowered_(rule)
    , graph_(lowered_.program_)
    , lhs_(roots(rule->lhs(), max_roots))
    , rhs_(roots(rule->rhs(), max_roots)) {
    if (lhs_.size() != rhs_.size()) {
        aggregate_rhs_ = rhs_.size() == 1 && Lit::isa(rule->rhs()->arity()) == lhs_.size();
        if (!aggregate_rhs_) throw std::invalid_argument("graph Rule needs one RHS for each LHS root");
    }
    for (auto lhs : lhs_)
        anchors_.push_back(graph_.anchor(lowered_.program_.capture(lowered_.lower(lhs))));
}

GraphMatches GraphRulePattern::find(const GraphView& graph, GraphMatchOptions options) const {
    return find_matches(graph_, graph, options);
}

GraphRulePlan
GraphRulePattern::plan(const GraphMatch& match, const GraphView& graph, const GuardContext* guards) const {
    if (!lowered_.program_.is_current() || !graph.is_current() || match.snapshot() != graph.snapshot())
        throw std::invalid_argument("graph Rule or match snapshot is stale");
    Def2Def bindings;
    for (auto [meta, capture] : lowered_.captures_)
        if (auto value = match.isa_capture(capture)) bindings.emplace(meta, value);
    auto rule = lowered_.rule_;
    if (rule->lhs() == rule->rhs()) return {MatchStatus::No_Match, {}, "identity graph Rule"};
    auto consistency = lowered_.validate_bindings(bindings);
    if (consistency != MatchStatus::Matched)
        return {consistency, {}, "whole telescope and projection bindings do not agree"};
    if (!lowered_.is_bound({rule->guard(), rule->rhs()}, bindings))
        return {MatchStatus::Unknown, {}, "guard or RHS contains unbound metavariables"};
    auto guard = evaluate_guard(rule, bindings, lhs_, match.anchors(), guards);
    if (guard.status != MatchStatus::Matched) return {guard.status, {}, guard.reason};

    auto plan = std::make_unique<RewritePlan>(match);
    DefMap<Recipe> built;
    size_t count                                    = 0;
    std::function<Recipe(const Def*, size_t)> build = [&](const Def* def, size_t depth) -> Recipe {
        if (depth >= 256 || ++count > 10000) throw std::invalid_argument("graph Rule recipe budget exceeded");
        if (auto i = built.find(def); i != built.end()) return i->second;
        auto result = [&]() -> Recipe {
            if (auto i = bindings.find(def); i != bindings.end()) return plan->import(i->second);
            if (auto literal = def->isa<Lit>()) return plan->literal(literal->type(), literal->flags());
            if (auto tuple = def->isa<Tuple>()) {
                std::vector<Recipe> args;
                for (auto op : tuple->ops())
                    args.push_back(build(op, depth + 1));
                return plan->tuple(std::move(args));
            }
            if (auto pack = def->isa_imm<Pack>()) {
                auto n = Lit::isa(*pack->shape());
                if (!n || *n > 10000 - std::min(count, size_t(10000)))
                    throw std::invalid_argument("graph Rule cannot expand this Pack recipe");
                auto elem = build(pack->body(), depth + 1);
                count += *n;
                return plan->tuple(std::vector<Recipe>(*n, elem));
            }
            if (auto extract = def->isa<Extract>()) {
                auto index = Lit::isa(extract->index());
                if (!index) throw std::invalid_argument("graph Rule needs a fixed result index");
                return plan->extract(build(extract->tuple(), depth + 1), *index);
            }
            if (auto app = def->isa<App>()) {
                auto callee = App::uncurry_callee(app)->isa<Axm>();
                if (!callee) throw std::invalid_argument("graph Rule recipes only call explicitly pure axioms");
                std::vector<const Def*> source;
                for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                    source.push_back(cur->arg());
                std::vector<Recipe> args;
                for (auto arg : source | std::views::reverse)
                    args.push_back(build(arg, depth + 1));
                return plan->call(callee, std::move(args));
            }
            if (graph.is_contained(def)) return plan->import(def);
            throw std::invalid_argument("unsupported graph Rule recipe value");
        }();
        built.emplace(def, result);
        return result;
    };
    try {
        size_t changed = 0;
        for (size_t i = 0; i != lhs_.size(); ++i) {
            if (!aggregate_rhs_ && lhs_[i] == rhs_[i]) continue;
            auto value = aggregate_rhs_ ? plan->extract(build(rhs_[0], 0), i) : build(rhs_[i], 0);
            plan->replace(match.at(anchors_[i]), value);
            ++changed;
        }
        if (!changed) return {MatchStatus::No_Match, {}, "identity graph Rule"};
    } catch (const std::invalid_argument& error) { return {MatchStatus::Unknown, {}, error.what()}; }
    return {MatchStatus::Matched, std::move(plan), {}};
}

} // namespace mim::pattern
