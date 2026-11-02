#include "mim/pattern/graph_rule.h"

#include <functional>
#include <stdexcept>

#include "mim/driver.h"
#include "mim/rewrite.h"
#include "mim/world.h"

#include "mim/pattern/guard.h"

namespace mim::pattern {

namespace {

bool literal_shape(const Def* shape) {
    if (shape->isa<Lit>()) return true;
    if (auto tuple = shape->isa<Tuple>())
        return std::ranges::all_of(tuple->ops(), [](auto d) { return d->template isa<Lit>(); });
    if (auto pack = shape->isa_imm<Pack>()) return Lit::isa(*pack->shape()) && pack->body()->isa<Lit>();
    return false;
}

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

bool GraphRulePattern::uses_graph_facts() const {
    auto& driver = lowered_.rule_->driver();
    DefVec pending{lowered_.rule_->guard(), lowered_.rule_->rhs()};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i >= 100000) return true;
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        if (auto axm = def->isa<Axm>()) {
            if (axm->sym() == driver.sym("compile.used_by") || axm->sym() == driver.sym("compile.only_used_by"))
                return true;
            if (driver.witness(axm->flags())) return true;
            continue;
        }
        if (def->isa<Var>()) continue;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return false;
}

GraphRulePlan
GraphRulePattern::plan(const GraphMatch& match, const GraphView& graph, const GuardContext* guards) const {
    if (!lowered_.program_.is_current() || !graph.is_current() || match.snapshot() != graph.snapshot())
        throw std::invalid_argument("graph Rule or match snapshot is stale");
    Def2Def bindings;
    for (auto [meta, capture] : lowered_.captures_)
        if (auto value = match.isa_capture(capture)) bindings.emplace(meta, value);
    auto rule = lowered_.rule_;
    lowered_.complete_bindings(bindings);
    if (!lowered_.binder_captures_safe(bindings))
        return {MatchStatus::Unknown, {}, "Rule capture would escape its matched binder"};
    if (rule->lhs() == rule->rhs()) return {MatchStatus::No_Match, {}, "identity graph Rule"};
    auto consistency = lowered_.validate_bindings(bindings);
    if (consistency != MatchStatus::Matched)
        return {consistency, {}, "whole telescope and projection bindings do not agree"};
    if (!lowered_.is_bound({rule->guard(), rule->rhs()}, bindings))
        return {MatchStatus::Unknown, {}, "guard or RHS contains unbound metavariables"};
    auto guard = evaluate_guard(rule, bindings, lhs_, match.anchors(), guards);
    if (guard.status != MatchStatus::Matched) return {guard.status, {}, guard.reason};

    auto plan = std::make_unique<RewritePlan>(match);
    for (auto [term, value] : guard.values)
        bindings.emplace(term, value);
    plan->witness_values(guard.values_world);
    if (!lowered_.binders_.empty()) {
        auto value = plan->instantiate(rule->rhs(), bindings);
        for (size_t i = 0; i != lhs_.size(); ++i)
            plan->replace(match.at(anchors_[i]), lhs_.size() == 1 ? value : plan->extract(value, i));
        return {MatchStatus::Matched, std::move(plan), {}};
    }
    DefMap<Recipe> built;
    size_t count                                    = 0;
    std::function<Recipe(const Def*, size_t)> build = [&](const Def* def, size_t depth) -> Recipe {
        if (depth >= 256 || ++count > 10000) throw std::invalid_argument("graph Rule recipe budget exceeded");
        if (auto i = built.find(def); i != built.end()) return i->second;
        auto result = [&]() -> Recipe {
            if (auto i = bindings.find(def); i != bindings.end()) {
                if (guard.values.contains(def)) return plan->witness_value(i->second);
                if (def == rule->lhs() && lowered_.is_meta(def)) return plan->instantiate(def, bindings);
                return plan->import(i->second);
            }
            if (lowered_.is_meta(def) && lowered_.can_bind(def, bindings)) {
                if (auto ex = def->isa<Extract>())
                    return plan->extract(build(ex->tuple(), depth + 1), Lit::as(ex->index()));
                std::vector<Recipe> fields;
                for (auto field : lowered_.fields_)
                    fields.push_back(build(field, depth + 1));
                return plan->tuple(std::move(fields));
            }
            if (def->isa<Lam>()) {
                DefVec pending{def};
                DefSet seen;
                for (size_t i = 0; i != pending.size(); ++i) {
                    if (++count > 10000) throw std::invalid_argument("graph Rule recipe budget exceeded");
                    auto term = pending[i];
                    if (!seen.insert(term).second || bindings.contains(term)) continue;
                    if (auto app = term->isa<App>()) {
                        auto axm = App::uncurry_callee(app)->isa<Axm>();
                        if (!axm || !graph.has_purity(axm->flags()))
                            throw std::invalid_argument("graph Rule RHS calls an operator without a purity guarantee");
                        for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                            pending.push_back(cur->arg());
                        continue;
                    }
                    if (term->isa<Var>() || term->isa<Axm>() || term->isa<Pi>()) continue;
                    for (auto op : term->ops())
                        if (op) pending.push_back(op);
                }
                return plan->instantiate(def, bindings);
            }
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
            if (auto app = def->isa<App>(); app && app->callee()->isa<Idx>() && def->is_closed())
                return plan->import(def);
            if (def->isa<Arr>() && def->is_closed()) {
                auto element = def;
                while (auto arr = element->isa<Arr>()) {
                    if (!literal_shape(*arr->shape())) break;
                    element = arr->body();
                }
                if (element->isa<Nat>() || element->isa<Axm>() || Idx::isa(element)) return plan->instantiate(def, {});
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
