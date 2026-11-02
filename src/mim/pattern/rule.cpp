#include "mim/pattern/rule.h"

#include <array>
#include <stdexcept>

#include <fe/restore.h>

#include "mim/driver.h"
#include "mim/rewrite.h"
#include "mim/tuple.h"
#include "mim/world.h"

#include "mim/pattern/guard.h"

namespace mim::pattern {

RulePattern::RulePattern(const Rule* rule)
    : rule_(rule) {
    if (!rule || !rule->is_set()) throw std::invalid_argument("pattern requires a fully defined Rule");
    auto count = Lit::isa(rule->dom()->arity());
    if (count && *count <= 10000) fields_.resize(*count);
    DefVec pending{rule->lhs(), rule->rhs(), rule->guard()};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i >= 100000) throw std::invalid_argument("Rule metavariable discovery budget exceeded");
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        if (auto ex = def->isa<Extract>(); ex && is_meta(ex)) {
            auto index = Lit::as(ex->index());
            if (index < fields_.size()) fields_[index] = ex;
            continue;
        }
        if (def->isa<Var>() || def->isa<Axm>()) continue;
        for (auto dep : def->deps())
            if (dep) pending.push_back(dep);
    }
    root_        = lower(rule->lhs());
    auto lambdas = [](const Def* root) {
        std::vector<std::pair<const Lam*, size_t>> result;
        std::vector<std::pair<const Def*, size_t>> pending{
            {root, 0}
        };
        DefSet seen;
        while (!pending.empty()) {
            auto [def, depth] = pending.back();
            pending.pop_back();
            if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
            if (auto lam = def->isa<Lam>()) {
                result.emplace_back(lam, depth);
                ++depth;
            }
            for (auto op : def->ops() | std::views::reverse)
                if (op) pending.emplace_back(op, depth);
        }
        return result;
    };
    auto lhs = lambdas(rule->lhs()), rhs = lambdas(rule->rhs());
    if (lhs.size() == rhs.size())
        for (size_t i = 0; i != lhs.size(); ++i)
            if (lhs[i].second == rhs[i].second && Check::test_alpha(lhs[i].first->type(), rhs[i].first->type()))
                binders_.emplace_back(lhs[i].first, rhs[i].first);
}

bool RulePattern::binder_captures_safe(const Def2Def& bindings) const {
    for (auto [meta, value] : bindings) {
        if (!is_meta(meta)) continue;
        for (auto [lhs, rhs] : binders_) {
            auto i      = bindings.find(lhs);
            auto source = i != bindings.end() ? i->second->isa<Lam>() : nullptr;
            auto var    = source ? source->has_var() : nullptr;
            if (!var || !value->has_free_var(var)) continue;
            DefVec pending{rule_->rhs()};
            DefSet seen;
            for (size_t j = 0; j != pending.size(); ++j) {
                auto term = pending[j];
                if (term == rhs) continue;
                if (term == meta) return false;
                if (!seen.insert(term).second || term->isa<Var>() || term->isa<Axm>()) continue;
                for (auto op : term->ops())
                    if (op) pending.push_back(op);
            }
        }
        for (auto var : value->free_vars())
            if (std::ranges::any_of(bindings,
                                    [&](const std::pair<const Def*, const Def*>& binding) {
                                        return binding.first->isa<Lam>() && binding.second->isa<Lam>()
                                            && binding.second->as<Lam>()->has_var() == var;
                                    })
                && !std::ranges::any_of(binders_, [&](const std::pair<const Lam*, const Lam*>& pair) {
                       auto i = bindings.find(pair.first);
                       return i != bindings.end() && i->second->isa<Lam>() && i->second->as<Lam>()->has_var() == var;
                   }))
                return false;
    }
    return true;
}

void RulePattern::map_binder(const Lam* source, Lam* target, Rewriter& rw, const Def2Def& bindings) const {
    for (auto [lhs, rhs] : binders_)
        if (rhs == source)
            if (auto i = bindings.find(lhs); i != bindings.end())
                if (auto matched = i->second->isa<Lam>())
                    if (auto var = matched->has_var()) {
                        rw.map(var, target->var());
                        if (auto old = lhs->has_var()) rw.map(old, target->var());
                    }
}

bool RulePattern::can_bind(const Def* def, const Def2Def& bindings) const {
    if (bindings.contains(def)) return true;
    if (auto ex = def->isa<Extract>()) return bindings.contains(ex->tuple());
    return !fields_.empty()
        && std::ranges::all_of(fields_, [&](const Def* field) { return field && bindings.contains(field); });
}

void RulePattern::complete_bindings(Def2Def& bindings) const {
    auto whole = rule_->has_var();
    auto i     = whole ? bindings.find(whole) : bindings.end();
    if (i == bindings.end()) return;
    auto value = i->second;
    for (size_t index = 0; index != fields_.size(); ++index) {
        auto field = fields_[index];
        if (!field || bindings.contains(field)) continue;
        if (auto tuple = value->isa<Tuple>(); tuple && index < tuple->num_ops())
            bindings.emplace(field, tuple->op(index));
        else if (auto pack = value->isa_imm<Pack>(); pack && Lit::isa(*pack->shape()) == fields_.size())
            bindings.emplace(field, pack->body());
    }
}

void RulePattern::bind_derived(Rewriter& rw, const Def2Def& bindings) const {
    auto whole = rule_->has_var();
    if (!whole) return;
    if (!bindings.contains(whole) && can_bind(whole, bindings)) {
        DefVec values;
        for (auto field : fields_)
            values.push_back(rw.lookup(field));
        rw.map(whole, rw.world().tuple(values));
    }
    if (auto value = rw.lookup(whole))
        for (size_t i = 0; i != fields_.size(); ++i)
            if (auto field = fields_[i]; field && !bindings.contains(field))
                rw.map(field, rw.world().extract(value, fields_.size(), i));
}

void RulePattern::check_definition() const {
    Def2Def available;
    DefVec pending{rule_->lhs()};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i >= 100000) rule_->blame("Rule metavariable discovery budget exceeded").bail();
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        if (is_meta(def)) {
            available.emplace(def, def);
            continue;
        }
        if (def->isa<Var>() || def->isa<Axm>()) continue;
        for (auto dep : def->deps())
            if (dep) pending.push_back(dep);
    }
    if (!is_bound({rule_->guard(), rule_->rhs()}, available))
        rule_->blame("Rule guard or RHS requires metavariables not bound by its LHS").bail();
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
    if (lowering_depth_ >= 256 || lowered_.size() >= 100000)
        throw std::invalid_argument("Rule pattern compilation budget exceeded");
    auto depth = fe::Restore(lowering_depth_, lowering_depth_ + 1);
    if (!def) throw std::invalid_argument("incomplete node in Rule pattern");
    if (auto i = lowered_.find(def); i != lowered_.end()) return i->second;
    if (def->is_closed() && (def->isa<Lam>() || (def->is_mutable() && def->type() && def->type()->isa<Type>()))) {
        auto value = program_.capture();
        auto alpha = program_.where(value, {value}, [def](std::span<const Def* const> values) {
            return Check::test_alpha(def, values[0]) ? CheckResult::True : CheckResult::False;
        });
        return program_.either(program_.exact(def), alpha);
    }
    if (auto app = def->isa<App>())
        if (auto var = App::uncurry_callee(app)->isa<Var>(); var && var->binder()->isa<Lam>())
            rule_->blame("Rule LHS bound function application requires higher-order matching").bail();

    if (auto lam = def->isa<Lam>()) {
        auto variable = program_.capture();
        if (auto var = lam->has_var()) captures_.emplace(var, variable);
        auto result  = program_.binder(variable, lower(lam->filter()), lower(lam->body()));
        result       = program_.all(result, program_.type(lower(lam->type())));
        auto capture = program_.capture(result);
        captures_.emplace(lam, capture);
        result = capture;
        lowered_.emplace(def, result);
        return result;
    }
    if (auto var = def->isa<Var>(); var && var->binder()->isa<Lam>())
        if (auto i = captures_.find(var); i != captures_.end()) return i->second;

    if (def->type() && def->type()->isa<Type>() && !def->is_closed() && !is_meta(def)) {
        DefVec pending{def}, metas;
        DefSet seen;
        for (size_t i = 0; i != pending.size(); ++i) {
            if (i >= 10000) throw std::invalid_argument("Rule dependent type traversal budget exceeded");
            auto term = pending[i];
            if (!seen.insert(term).second) continue;
            if (is_meta(term)) {
                metas.push_back(term);
                continue;
            }
            if (term->isa<Var>() || term->isa<Axm>()) continue;
            for (auto dep : term->deps())
                if (dep) pending.push_back(dep);
        }
        if (!metas.empty()) {
            auto value = program_.capture();
            std::vector<Pattern> inputs{value};
            for (auto meta : metas) {
                lower(meta);
                inputs.push_back(captures_.at(meta));
            }
            auto result = program_.where(value, std::move(inputs), [def, metas](std::span<const Def* const> values) {
                auto scratch = def->world().inherit();
                Rewriter importer(*scratch);
                for (size_t i = 0; i != metas.size(); ++i)
                    importer.map(metas[i], importer.rewrite(values[i + 1]));
                auto expected = importer.rewrite(def);
                auto actual   = importer.rewrite(values[0]);
                return Check::test_alpha(expected, actual) ? CheckResult::True : CheckResult::False;
            });
            lowered_.emplace(def, result);
            return result;
        }
    }

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
    if (auto app = def->isa<App>();
        app && app->axm() && app->curry() == 0 && def->driver().graph_operations().is_commutative(app->axm()->flags()))
        if (auto pair = app->arg()->isa<Tuple>(); pair && pair->num_ops() == 2) {
            auto swapped = program_.node(Node::Tuple, {lower(pair->op(1)), lower(pair->op(0))}, pair->flags());
            swapped      = program_.all(swapped, program_.type(lower(pair->type())));
            operands[1]  = program_.either(operands[1], swapped);
        }
    auto result = program_.node(def->node(), operands, def->flags());
    if (def->isa<Tuple>() && !operands.empty()) {
        auto body = operands.front();
        for (size_t i = 1; i != operands.size(); ++i)
            body = program_.all(body, operands[i]);
        auto pack = program_.node(Node::Pack, {program_.literal(def->world().type_nat(), operands.size()), body});
        result    = program_.either(result, pack);
    }
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
    complete_bindings(bindings);
    if (!binder_captures_safe(bindings)) return {MatchStatus::Unknown, {}, outcome.steps};
    auto status = validate_bindings(bindings);
    if (status != MatchStatus::Matched) return {status, {}, outcome.steps};
    return {outcome.status, std::move(bindings), outcome.steps};
}

MatchStatus RulePattern::validate_bindings(const Def2Def& bindings) const {
    // A whole telescope and one of its projections must describe the same substitution.
    // Inspect existing tuple/pack nodes or projections; never construct an Extract in the source.
    const Def* whole = nullptr;
    for (auto [meta, value] : bindings)
        if (auto var = meta->isa<Var>(); var && var->binder() == rule_) whole = value;
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
                if (!expected) return MatchStatus::Unknown;
                if (expected != value) return MatchStatus::No_Match;
            }
    return MatchStatus::Matched;
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
            if (!can_bind(def, bindings)) return false;
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

RuleReplacement RulePattern::build_replacement(const Def* candidate,
                                               Rewriter& context,
                                               MatchOptions options,
                                               const GuardContext* guards) const {
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

    class BinderImporter : public Rewriter {
    public:
        BinderImporter(World& world, const RulePattern& pattern, const Def2Def& bindings)
            : Rewriter(world)
            , pattern_(pattern)
            , bindings_(bindings) {}
        const Def* rewrite_mut_Lam(Lam* lam) override {
            auto fresh = world().mut_lam(rewrite(lam->type())->as<Pi>());
            pattern_.map_binder(lam, fresh, *this, bindings_);
            return rewrite_stub(lam, fresh);
        }
        const Def* lookup(const Def* def) override {
            if (auto value = Rewriter::lookup(def)) return value;
            if (auto i = bindings_.find(def); i != bindings_.end() && pattern_.is_meta(def))
                return map(def, rewrite(i->second));
            return nullptr;
        }

    private:
        const RulePattern& pattern_;
        const Def2Def& bindings_;
    };
    BinderImporter rw(target, *this, result.bindings);
    for (auto var : vars)
        rw.map(var, context.rewrite(var));
    for (auto [meta, value] : result.bindings)
        if (is_meta(meta) && !value->has_free_vars())
            rw.map(meta, context.rewrite(value));
        else if (is_meta(meta)) {
            bool bound = std::ranges::any_of(result.bindings, [&](const std::pair<const Def*, const Def*>& binding) {
                auto lam = binding.first->isa<Lam>() ? binding.second->isa<Lam>() : nullptr;
                return lam && lam->has_var() && value->has_free_var(lam->has_var());
            });
            if (!bound) rw.map(meta, context.rewrite(value));
        }
    if (binders_.empty()) bind_derived(rw, result.bindings);
    std::array lhs{rule_->lhs()}, matched{candidate};
    auto guard = evaluate_guard(rule_, result.bindings, lhs, matched, guards);
    if (guard.status != MatchStatus::Matched) return {guard.status, nullptr, result.steps, guard.reason};
    Rewriter witness_importer(target);
    for (auto [term, value] : guard.values)
        rw.map(term, witness_importer.rewrite(value));
    auto replacement = rw.rewrite(rule_->rhs());
    auto expected    = candidate->type() ? context.rewrite(candidate->type()) : nullptr;
    auto actual      = replacement->type();
    if (actual != expected && (!actual || !expected || !Check::test_alpha(actual, expected)))
        return {MatchStatus::Unknown, nullptr, result.steps, "RHS does not preserve the matched type"};
    return {MatchStatus::Matched, replacement, result.steps, {}};
}

} // namespace mim::pattern
