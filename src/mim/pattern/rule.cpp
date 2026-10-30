#include "mim/pattern/rule.h"

#include <stdexcept>

#include "mim/rewrite.h"
#include "mim/tuple.h"
#include "mim/world.h"

namespace mim::pattern {

RulePattern::RulePattern(const Rule* rule)
    : rule_(rule) {
    if (!rule || !rule->is_set()) throw std::invalid_argument("pattern requires a fully defined Rule");
    root_ = lower(rule->lhs());
}

bool RulePattern::is_meta(const Def* def) const {
    if (auto var = def->isa<Var>()) return var->binder() == rule_;
    if (auto extract = def->isa<Extract>())
        if (auto var = extract->tuple()->isa<Var>(); var && var->binder() == rule_) {
            if (!extract->index()->isa<Lit>())
                throw std::invalid_argument("rule capture requires a literal projection index");
            return true;
        }
    return false;
}

Pattern RulePattern::lower(const Def* def) {
    if (!def) throw std::invalid_argument("incomplete node in Rule pattern");
    if (auto i = lowered_.find(def); i != lowered_.end()) return i->second;

    if (is_meta(def)) {
        // Register the capture before its type: dependent types can refer to other captures.
        if (auto i = captures_.find(def); i != captures_.end()) return i->second;
        auto capture = program_.capture();
        captures_.emplace(def, capture);
        auto result = def->type() ? program_.all(capture, program_.type(lower(def->type()))) : capture;
        lowered_.emplace(def, result);
        return result;
    }

    // Literal and ground-type equality use canonical identity within this World.
    // Open terms are lowered structurally so their Rule metavariables can bind.
    if (def->is_mutable() || def->node() == Node::Var || def->is_ground()) {
        auto result = program_.exact(def);
        lowered_.emplace(def, result);
        return result;
    }

    auto operands = std::vector<Pattern>();
    for (auto op : def->ops())
        operands.push_back(lower(op));
    auto result = program_.node(def->node(), std::move(operands), def->flags());
    if (auto type = def->type()) result = program_.all(result, program_.type(lower(type)));
    lowered_.emplace(def, result);
    return result;
}

RuleMatch RulePattern::match_lhs(const Def* candidate, MatchOptions options) const {
    if (!program_.is_current()) throw std::invalid_argument("Rule pattern references a rebuilt World");
    if (!candidate) return {MatchStatus::No_Match, {}, 0};
    if (&candidate->world() != &rule_->world()) throw std::invalid_argument("Rule matching requires the source World");
    if (Rule::is_in_rule(candidate)) return {MatchStatus::No_Match, {}, 0};
    auto outcome  = match(program_, *root_, candidate, options);
    auto bindings = Def2Def();
    if (outcome.result)
        for (auto [meta, capture] : captures_)
            if (auto value = outcome.result->isa_capture(capture)) bindings.emplace(meta, value);
    // A whole telescope and one of its projections must describe the same substitution.
    // Inspect existing tuple/pack nodes or projections; never construct an Extract in the source.
    const Def* whole = nullptr;
    for (auto [meta, value] : bindings)
        if (meta->isa<Var>()) whole = value;
    if (whole)
        for (auto [meta, value] : bindings)
            if (auto projection = meta->isa<Extract>()) {
                auto index          = Lit::as(projection->index());
                const Def* expected = nullptr;
                if (auto tuple = whole->isa<Tuple>(); tuple && index < tuple->num_ops())
                    expected = tuple->op(index);
                else if (auto pack = whole->isa_imm<Pack>(); pack && Lit::isa(*pack->shape()).value_or(0) > index)
                    expected = pack->body();
                else if (auto extract = value->isa<Extract>();
                         extract && extract->tuple() == whole && Lit::isa(extract->index()) == index)
                    expected = value;
                if (!expected) return {MatchStatus::Unknown, {}, outcome.steps};
                if (expected != value) return {MatchStatus::No_Match, {}, outcome.steps};
            }
    return {outcome.status, std::move(bindings), outcome.steps};
}

bool RulePattern::is_bound(Defs roots, const Def2Def& bindings) const {
    auto pending = DefVec();
    for (auto root : roots)
        pending.push_back(root);
    auto seen = DefSet();
    while (!pending.empty()) {
        auto def = pending.back();
        pending.pop_back();
        if (!def || !seen.insert(def).second) continue;
        if (is_meta(def)) {
            if (!bindings.contains(def)) return false;
            continue;
        }
        if (def->node() == Node::Hole || !def->is_set()) return false;
        // Inspect the binder instead of asking for a sequence Var's type in the source World.
        if (auto var = def->isa<Var>()) {
            if (var->binder()->node() == Node::Rule) return false;
            pending.push_back(var->binder());
        } else {
            for (auto dep : def->deps())
                pending.push_back(dep);
        }
    }
    return true;
}

RuleReplacement RulePattern::build_replacement(const Def* candidate, World& target, MatchOptions options) const {
    Rewriter context(target);
    return build_replacement(candidate, context, options);
}

RuleReplacement RulePattern::build_replacement(const Def* candidate, Rewriter& context, MatchOptions options) const {
    if (!program_.is_current()) throw std::invalid_argument("Rule pattern references a rebuilt World");
    auto& target = context.world();
    if (&target == &rule_->world() || &target.driver() != &rule_->driver())
        throw std::invalid_argument("rule replacement requires a separate World of the same Driver");
    auto result = match_lhs(candidate, options);
    if (result.status != MatchStatus::Matched) return {result.status, nullptr, result.steps, "LHS did not match"};
    auto vars = candidate->free_vars();
    for (auto var : vars)
        if (!context.lookup(var) && !context.lookup(var->binder()))
            return {MatchStatus::Unknown, nullptr, result.steps, "open candidate requires a binder mapping"};
    for (auto term : {rule_->guard(), rule_->rhs()})
        for (auto var : term->free_vars())
            if (var->binder() != rule_ && !vars.contains(var))
                return {MatchStatus::Unknown, nullptr, result.steps, "open replacement requires a binder mapping"};
    if (!is_bound({rule_->guard(), rule_->rhs()}, result.bindings))
        return {MatchStatus::Unknown, nullptr, result.steps, "guard or RHS contains an unbound metavariable or Hole"};

    Rewriter rw(target);
    for (auto var : vars)
        rw.map(var, context.rewrite(var));
    for (auto [meta, value] : result.bindings)
        rw.map(meta, context.rewrite(value));
    auto guard = rw.rewrite(rule_->guard());
    if (guard == target.lit_ff()) return {MatchStatus::No_Match, nullptr, result.steps, "guard is false"};
    if (guard != target.lit_tt())
        return {MatchStatus::Unknown, nullptr, result.steps, "guard did not normalize to true"};
    auto replacement = rw.rewrite(rule_->rhs());
    auto expected    = candidate->type() ? context.rewrite(candidate->type()) : nullptr;
    auto actual      = replacement->type();
    if (actual != expected && (!actual || !expected || !Check::test_alpha(actual, expected)))
        return {MatchStatus::Unknown, nullptr, result.steps, "RHS does not preserve the matched type"};
    return {MatchStatus::Matched, replacement, result.steps, {}};
}

} // namespace mim::pattern
