#include "mim/pattern/rewrite.h"

#include <stdexcept>

#include "mim/rewrite.h"
#include "mim/world.h"

#include "mim/pattern/rule.h"
#include "mim/pattern/witness.h"

namespace mim::pattern {

RewritePlan::RewritePlan(const GraphMatch& match)
    : snapshot_(match.snapshot())
    , reads_(match.reads().begin(), match.reads().end()) {
    if (!snapshot_->is_live()) throw std::invalid_argument("match snapshot has already been published");
    for (auto def : match.nodes())
        matched_.insert(def);
}

void RewritePlan::check(Recipe recipe) const {
    if (recipe.plan_ != this || recipe.index_ >= entries_.size()) throw std::invalid_argument("foreign recipe");
}

Recipe RewritePlan::append(Entry entry) {
    for (auto arg : entry.args)
        check(arg);
    auto result = Recipe(this, entries_.size());
    entries_.push_back(std::move(entry));
    return result;
}

Recipe RewritePlan::import(const Def* def) {
    if (!def) throw std::invalid_argument("recipe import must not be null");
    return append({.kind = Kind::Import, .def = def});
}

Recipe RewritePlan::literal(const Def* type, flags_t bits) {
    if (!type) throw std::invalid_argument("literal recipe requires a type");
    return append({.kind = Kind::Literal, .def = type, .bits = bits});
}

Recipe RewritePlan::call(const Axm* callee, std::vector<Recipe> args) {
    if (!callee || args.empty()) throw std::invalid_argument("call recipe requires an axiom and arguments");
    return append({.kind = Kind::Call, .def = callee, .args = std::move(args)});
}

Recipe RewritePlan::tuple(std::vector<Recipe> args) { return append({.kind = Kind::Tuple, .args = std::move(args)}); }

Recipe RewritePlan::extract(Recipe tuple, size_t index) {
    return append({.kind = Kind::Extract, .bits = index, .args = {tuple}});
}

Recipe RewritePlan::instantiate(const Def* term, const Def2Def& bindings) {
    return append({.kind = Kind::Instantiate, .def = term, .bindings = bindings});
}

Recipe RewritePlan::witness_value(const Def* value) {
    if (!witness_world_ || &value->world() != witness_world_.get() || !is_witness_data(value))
        throw std::invalid_argument("witness recipe requires owned frozen literal data");
    return append({.kind = Kind::Witness, .def = value});
}

void RewritePlan::replace(const Def* old, Recipe recipe) {
    check(recipe);
    if (!old || !matched_.contains(old) || !replacements_.emplace(old, recipe.index_).second)
        throw std::invalid_argument("replacement must name a distinct matched node");
}

namespace {

struct RebuildBudget {
    RewriteOptions options;
    size_t work = 0;
    void touch() {
        if (++work > options.max_source_nodes) throw RebuildLimit("graph rebuild source work budget exceeded");
    }
};

class GraphImporter : public DAGRewriter {
public:
    GraphImporter(const GraphView& graph, World& target, RebuildBudget& budget)
        : DAGRewriter(target, budget.options.max_source_nodes)
        , graph_(graph)
        , budget_(budget) {}
    const Def* rewrite(const Def* def) override {
        budget_.touch();
        return DAGRewriter::rewrite(def);
    }
    const Def* lookup(const Def* def) override {
        if (auto mapped = Rewriter::lookup(def)) return mapped;
        // Locally constructed test/client axioms use the global plugin ID, which is not an annex.
        if (auto axm = def->isa<Axm>(); axm && axm->plugin() == Annex::Global_Plugin) return nullptr;
        if (auto flags = graph_.world().annexes().opaque(def)) return map(def, world().annex(*flags));
        return nullptr;
    }

private:
    const GraphView& graph_;
    RebuildBudget& budget_;
};

class RHSImporter : public GraphImporter {
public:
    RHSImporter(const GraphView& graph, World& target, RebuildBudget& budget)
        : GraphImporter(graph, target, budget)
        , graph_(graph) {}
    const RulePattern* pattern = nullptr;
    const Def2Def* captures    = nullptr;
    const Def* rewrite_mut_Lam(Lam* lam) override {
        auto fresh = world().mut_lam(rewrite(lam->type())->as<Pi>());
        if (pattern && captures) pattern->map_binder(lam, fresh, *this, *captures);
        return rewrite_stub(lam, fresh);
    }
    const Def* lookup(const Def* def) override {
        if (auto value = GraphImporter::lookup(def)) return value;
        if (captures)
            if (auto i = captures->find(def); i != captures->end() && pattern && pattern->is_capture(def))
                return map(def, rewrite(i->second));
        return nullptr;
    }

    const Def* rewrite_imm_App(const App* app) override {
        auto axm = App::uncurry_callee(app)->isa<Axm>();
        if (!axm && app->type()->isa<Type>()) return Rewriter::rewrite_imm_App(app);
        if (!axm) throw InstantiationFailure("graph Rule RHS may only introduce axiom calls");
        std::vector<const Def*> source;
        for (auto cur = app; cur; cur = cur->callee()->isa<App>())
            source.push_back(cur->arg());
        DefVec args;
        for (auto arg : source | std::views::reverse)
            args.push_back(rewrite(arg));
        if (!graph_.is_pure_call(axm->flags(), args))
            throw InstantiationFailure("graph Rule RHS call has no satisfied purity guarantee");
        auto callee = rewrite(axm);
        auto value  = callee;
        for (size_t i = 0; i != args.size(); ++i) {
            auto prefix = value;
            for (size_t j = i; j-- != 0;) {
                auto part = prefix->isa<App>();
                if (!part || part->arg() != args[j]) {
                    prefix = nullptr;
                    break;
                }
                prefix = part->callee();
            }
            if (prefix != callee)
                throw InstantiationFailure("graph Rule RHS normalization changed its application prefix");
            value = world().app(value, args[i]);
        }
        return value;
    }

    DefVec dependencies(const Def* def) const override {
        if (auto app = def->isa<App>()) {
            if (auto axm = App::uncurry_callee(app)->isa<Axm>()) {
                DefVec result{app->type(), axm};
                for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                    result.push_back(cur->arg());
                return result;
            }
        }
        return DAGRewriter::dependencies(def);
    }

private:
    const GraphView& graph_;
};

} // namespace

class GraphRebuilder : public DAGRewriter {
public:
    bool descend(const Def* def) const override { return !replacements_.contains(def); }
    GraphRebuilder(const GraphView& graph,
                   std::span<const RewritePlan* const> plans,
                   World& target,
                   RewriteOptions options)
        : DAGRewriter(target, options.max_source_nodes)
        , graph_(graph)
        , budget_{options}
        , importer_(graph, target, budget_)
        , options_(options) {
        if (!graph.is_current()) throw std::invalid_argument("graph snapshot is stale");
        size_t count = 0;
        for (auto plan : plans) {
            if (!plan || plan->snapshot_ != graph.snapshot() || plan->replacements_.empty())
                throw std::invalid_argument("plan does not belong to this snapshot or has no replacements");
            if (plan->witness_world_ && &plan->witness_world_->driver() != &graph.world().driver())
                throw std::invalid_argument("witness values belong to another Driver");
            if (plan->entries_.size() > options.max_recipes - std::min(count, options.max_recipes))
                throw RebuildLimit("replacement recipe budget exceeded");
            count += plan->entries_.size();
            memo_.emplace(plan, std::vector<const Def*>(plan->entries_.size(), nullptr));
            for (auto [def, recipe] : plan->replacements_)
                if (!graph.is_pure(def) || !replacements_.emplace(def, std::pair{plan, recipe}).second)
                    throw std::invalid_argument("replacement is impure, unsupported, or overlaps another plan");
            for (const auto& entry : plan->entries_) {
                if (entry.def && &entry.def->world() != &graph.world() && entry.kind != RewritePlan::Kind::Witness)
                    throw std::invalid_argument("recipe references another World");
                if (entry.kind == RewritePlan::Kind::Witness
                    && (!plan->witness_world_ || &entry.def->world() != plan->witness_world_.get()
                        || !is_witness_data(entry.def)))
                    throw std::invalid_argument("recipe has an invalid frozen witness value");
                if (entry.kind == RewritePlan::Kind::Import && !graph.is_contained(entry.def))
                    throw std::invalid_argument("recipe imports a value outside the graph");
                if (entry.kind == RewritePlan::Kind::Call && !graph.has_purity(entry.def->flags()))
                    throw std::invalid_argument("recipe calls an operator without a purity guarantee");
                if (entry.kind == RewritePlan::Kind::Literal
                    && (!entry.def->is_closed() || entry.def->isa<Sigma>() || entry.def->isa<Arr>()
                        || entry.def->isa<Pi>()))
                    throw std::invalid_argument("literal recipes require a closed scalar type");
                if (entry.kind == RewritePlan::Kind::Instantiate) {
                    for (auto [meta, value] : entry.bindings) {
                        if (!meta || &meta->world() != &graph.world())
                            throw std::invalid_argument("RHS instantiation has foreign capture bindings");
                        if (meta->isa<Lam>() || (meta->isa<Var>() && meta->as<Var>()->binder()->isa<Lam>())) continue;
                        if (plan->witness_world_ && &value->world() == plan->witness_world_.get()
                            && is_witness_data(value))
                            continue;
                        if (!graph.is_contained(value) && !value->has_free_vars())
                            throw std::invalid_argument("RHS instantiation has foreign capture bindings");
                        DefVec pending{value};
                        DefSet seen;
                        for (size_t i = 0; i != pending.size(); ++i) {
                            budget_.touch();
                            auto source = pending[i];
                            if (!seen.insert(source).second) continue;
                            if (auto app = source->isa<App>()) {
                                if (app->callee()->isa<Idx>()) continue;
                                auto axm = App::uncurry_callee(app)->isa<Axm>();
                                DefVec args;
                                for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                                    args.push_back(cur->arg());
                                std::ranges::reverse(args);
                                if (!axm || !graph.is_pure_call(axm->flags(), args))
                                    throw InstantiationFailure("RHS binder capture includes an unproved computation");
                                for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                                    pending.push_back(cur->arg());
                            } else if (!source->is_mutable() && !source->isa<Var>() && !source->isa<Axm>()) {
                                for (auto op : source->ops())
                                    if (op) pending.push_back(op);
                            }
                        }
                    }
                }
            }
        }
        for (auto a : plans)
            for (auto b : plans)
                if (a != b)
                    for (auto [write, recipe] : a->replacements_)
                        if (std::ranges::find(b->reads_, write) != b->reads_.end())
                            throw std::invalid_argument("plans have a read-write conflict");

        // Recipe imports use the simultaneous replacement environment. Include dependencies hidden
        // inside imported immutable terms; checking only direct import handles would miss cycles.
        DefMap<std::vector<const Def*>> dependencies;
        size_t scanned = 0;
        for (auto [def, replacement] : replacements_) {
            const auto [plan, recipe] = replacement;
            auto pending              = std::vector<size_t>{recipe};
            auto seen                 = std::vector<bool>(plan->entries_.size());
            for (size_t i = 0; i != pending.size(); ++i) {
                auto index = pending[i];
                if (seen[index]) continue;
                seen[index]       = true;
                const auto& entry = plan->entries_[index];
                for (auto arg : entry.args)
                    pending.push_back(arg.index_);
                if (entry.kind != RewritePlan::Kind::Import) continue;
                auto defs = std::vector<const Def*>{entry.def};
                DefSet visited;
                for (size_t j = 0; j != defs.size(); ++j) {
                    auto current = defs[j];
                    if (!visited.insert(current).second) continue;
                    if (++scanned > options.max_source_nodes) throw RebuildLimit("source dependency budget exceeded");
                    if (replacements_.contains(current)) {
                        dependencies[def].push_back(current);
                        continue; // Its new recipe, not its old operands, determines dependencies.
                    }
                    if (current->is_mutable() || current->node() == Node::Var || current->node() == Node::Lam
                        || current->node() == Node::Pi || current->node() == Node::Axm)
                        continue;
                    for (auto dep : current->deps())
                        if (dep) defs.push_back(dep);
                }
            }
        }
        DefMap<unsigned> state;
        std::function<void(const Def*, size_t)> visit = [&](const Def* def, size_t depth) {
            if (state[def] == 1) throw std::invalid_argument("replacement dependency cycle");
            if (state[def] == 2) return;
            if (depth >= options.max_depth) throw RebuildLimit("replacement dependency depth exceeded");
            state[def] = 1;
            for (auto next : dependencies[def])
                visit(next, depth + 1);
            state[def] = 2;
        };
        for (auto [def, replacement] : replacements_)
            visit(def, 0);
        // A decoded pure operation proves its own curry spine, not evaluated input applications.
        DefSet pure;
        for (auto def : graph.operations())
            if (graph.is_pure(def))
                for (auto cur = def; auto app = cur->isa<App>(); cur = app->callee()) {
                    budget_.touch();
                    pure.insert(cur);
                }
        for (auto def : graph.nodes())
            if (def->isa<App>() && !pure.contains(def)) protected_.insert(def);
        DefVec pending(graph.nodes().begin(), graph.nodes().end());
        DefSet seen;
        DefSet local_lambdas;
        for (size_t i = 0; i != pending.size(); ++i) {
            budget_.touch();
            auto def = pending[i];
            if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
            if (auto lam = def->isa<Lam>()) {
                if (lam->is_external()) continue;
                local_lambdas.insert(lam);
            }
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        seen.clear();
        for (size_t i = 0; i != pending.size(); ++i) {
            budget_.touch();
            auto def = pending[i];
            if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
            if (auto lam = def->isa<Lam>(); lam && lam->is_external()) continue;
            if (auto app = def->isa<App>()) {
                auto callee = App::uncurry_callee(app);
                auto axm    = callee->isa<Axm>();
                auto var    = callee->isa<Var>();
                DefVec args;
                for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                    args.push_back(cur->arg());
                std::ranges::reverse(args);
                if ((axm && graph.is_pure_call(axm->flags(), args)) || (var && local_lambdas.contains(var->binder())))
                    for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                        pure.insert(cur);
                else if (!pure.contains(def))
                    protected_.insert(def);
            }
        }
        for (auto def : pure)
            protected_.erase(def);
    }

    const Def* rewrite_ready(const Def* def) override {
        budget_.touch();
        auto replacement = replacements_.find(def);
        if (replacement == replacements_.end()) {
            auto value = Rewriter::rewrite(def);
            if (protected_.contains(def)) retained_.insert_or_assign(def, value);
            return value;
        }
        if (auto i = results_.find(def); i != results_.end()) return i->second;
        auto value    = evaluate(replacement->second.first, replacement->second.second, 0);
        auto expected = importer_.rewrite(def->type());
        if (value->type() != expected && (!value->type() || !Check::test_alpha(value->type(), expected)))
            throw InstantiationFailure("replacement does not preserve the source type");
        for (auto var : value->free_vars())
            if (!environment_.contains(var)) throw InstantiationFailure("replacement escapes the selected scope");
        results_.emplace(def, value);
        return map(def, value);
    }

    const Def* rewrite_mut(Def* def) override { return map(def, importer_.rewrite(def)); }
    const Def* rewrite_imm_Lam(const Lam* def) override { return importer_.rewrite(def); }
    const Def* rewrite_imm_Pi(const Pi* def) override { return importer_.rewrite(def); }

    GraphRewrite build(std::unique_ptr<World> target) {
        bool changed = false;
        if (auto scope = graph_.scope()) {
            auto pi    = importer_.rewrite(scope->type())->as<Pi>();
            new_scope_ = world().mut_lam(pi)->set(scope->dbg_key());
            importer_.map(scope, new_scope_);
            map(scope, new_scope_);
            if (auto var = scope->has_var()) {
                new_scope_->var()->set(var->dbg_key());
                importer_.map(var, new_scope_->var());
                map(var, new_scope_->var());
                environment_.insert(new_scope_->var());
                bindings_.emplace_back(var, new_scope_->var());
            }
            auto filter = importer_.rewrite(scope->filter());
            // Both bodies see the same unset scope stub, so self-applications cannot unfold one side.
            GraphImporter original_importer(graph_, world(), budget_);
            original_importer.map(scope, new_scope_);
            for (auto [old, value] : bindings_)
                original_importer.map(old, value);
            auto original = original_importer.rewrite(scope->body());
            auto body     = graph_.region_path().empty() ? rewrite(scope->body()) : rewrite_path(scope->body(), 0);
            if (!graph_.region_path().empty() && graph_.shared_region() && region_changed_) {
                GraphImporter shared(graph_, world(), budget_);
                shared.map(scope, new_scope_);
                if (auto var = scope->has_var()) shared.map(var, new_scope_->var());
                for (auto [old, fresh] : region_lams_)
                    shared.map(old, fresh);
                body = shared.rewrite(scope->body());
            }
            changed = graph_.region_path().empty() ? body != original : region_changed_;
            new_scope_->set(filter, body);
            if (scope->is_external()) new_scope_->externalize();
            // Other functions use a separate map: a hash-consed value shared across scopes must
            // retain its original implementation outside the selected function body.
            for (auto ext : graph_.world().externals().muts())
                if (ext != scope)
                    if (auto copy = original_importer.rewrite(ext)->isa_mut(); copy && !copy->is_external())
                        copy->externalize();
        } else {
            for (auto root : graph_.roots())
                if (root->has_free_vars())
                    throw std::invalid_argument("open graph rewrite requires an explicit function scope");
        }
        auto roots = std::vector<const Def*>();
        for (auto root : graph_.roots()) {
            auto original = new_scope_ ? nullptr : importer_.rewrite(root);
            auto value    = graph_.region_path().empty() ? rewrite(root) : lookup(root);
            roots.push_back(value);
            if (!new_scope_) changed |= value != original;
        }
        // A pure replacement key may still enclose an unproved computation. Require that each
        // such computation survives in the selected result, including after normalization.
        DefSet reachable;
        auto pending = new_scope_ ? std::vector<const Def*>{new_scope_->body()} : roots;
        for (size_t i = 0; i != pending.size(); ++i) {
            budget_.touch();
            auto def = pending[i];
            if (!reachable.insert(def).second) continue;
            if (auto lam = def->isa<Lam>(); lam && lam->is_set() && !lam->is_external()) {
                pending.push_back(lam->body());
                continue;
            }
            if (def->is_mutable() || def->node() == Node::Lam || def->node() == Node::Pi || def->node() == Node::Var
                || def->node() == Node::Axm)
                continue;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        for (auto def : protected_) {
            auto i = retained_.find(def);
            if (i == retained_.end())
                if (auto value = importer_.lookup(def)) i = retained_.emplace(def, value).first;
            if (i == retained_.end() || !reachable.contains(i->second))
                throw PurityFailure("replacement removes a computation without a purity proof");
        }
        world().verify();
        if (!graph_.is_current()) throw std::invalid_argument("graph changed while building replacements");
        return GraphRewrite(&graph_.world(), std::move(target), graph_.snapshot(), std::move(roots), new_scope_,
                            replacements_.size(), changed);
    }

private:
    const Def* rewrite_path(const Def* def, size_t position) {
        struct Frame {
            const Def* source;
            size_t index;
            Lam* fresh;
            const Def* type;
            DefVec ops;
        };
        std::vector<Frame> frames;
        for (; position != graph_.region_path().size(); ++position) {
            budget_.touch();
            auto index = graph_.region_path()[position];
            GraphImporter preserve(graph_, world(), budget_);
            preserve.map(graph_.scope(), new_scope_);
            for (auto [old, value] : bindings_)
                preserve.map(old, value);
            if (auto lam = def->isa<Lam>()) {
                if (index != 1 || lam->is_external()) throw std::invalid_argument("unsupported region lambda path");
                auto fresh = world().mut_lam(importer_.rewrite(lam->type())->as<Pi>())->set(lam->dbg_key());
                if (auto var = lam->has_var()) {
                    fresh->var()->set(var->dbg_key());
                    importer_.map(var, fresh->var());
                    map(var, fresh->var());
                    environment_.insert(fresh->var());
                    bindings_.emplace_back(var, fresh->var());
                }
                importer_.map(lam, fresh);
                map(lam, fresh);
                region_lams_.emplace_back(lam, fresh);
                frames.push_back({
                    def,
                    index,
                    fresh,
                    nullptr,
                    {importer_.rewrite(lam->filter()), nullptr}
                });
            } else {
                if (!def->isa<App>() && !def->isa<Tuple>() && !def->isa_imm<Pack>())
                    throw std::invalid_argument("unsupported region occurrence owner");
                DefVec ops;
                for (size_t i = 0; i != def->num_ops(); ++i)
                    ops.push_back(i == index ? nullptr : preserve.rewrite(def->op(i)));
                auto type = def->isa_imm<Pack>() ? preserve.rewrite(*def->as<Pack>()->shape())
                                                 : preserve.rewrite(def->type());
                frames.push_back({def, index, nullptr, type, std::move(ops)});
            }
            def = def->op(index);
        }
        auto original   = importer_.rewrite(def);
        auto value      = rewrite(def);
        region_changed_ = value != original;
        for (auto& frame : frames | std::views::reverse) {
            frame.ops[frame.index] = value;
            if (frame.fresh)
                value = frame.fresh->set(frame.ops[0], frame.ops[1]);
            else if (frame.source->isa<App>())
                value = world().app(frame.ops[0], frame.ops[1]);
            else if (frame.source->isa<Tuple>())
                value = world().tuple(frame.type, frame.ops);
            else
                value = world().seq(true, frame.type, frame.ops.back());
        }
        return value;
    }

    const Def* evaluate(const RewritePlan* plan, size_t index, size_t depth) {
        if (depth >= options_.max_depth) throw RebuildLimit("recipe evaluation depth exceeded");
        auto& value = memo_.at(plan)[index];
        if (value) return value;
        const auto& entry = plan->entries_[index];
        auto args         = DefVec();
        for (auto arg : entry.args)
            args.push_back(evaluate(plan, arg.index_, depth + 1));
        switch (entry.kind) {
            case RewritePlan::Kind::Witness: value = importer_.rewrite(entry.def); break;
            case RewritePlan::Kind::Import: value = rewrite(entry.def); break;
            case RewritePlan::Kind::Literal: value = world().lit(importer_.rewrite(entry.def), entry.bits); break;
            case RewritePlan::Kind::Call: {
                if (!graph_.is_pure_call(entry.def->flags(), {args.data(), args.size()}))
                    throw std::invalid_argument("recipe arguments do not satisfy the operator purity proof");
                auto callee = importer_.rewrite(entry.def);
                value       = callee;
                for (size_t i = 0; i != args.size(); ++i) {
                    budget_.touch();
                    auto prefix = value;
                    for (size_t j = i; j-- != 0;) {
                        budget_.touch();
                        auto app = prefix->isa<App>();
                        if (!app || app->arg() != args[j]) {
                            prefix = nullptr;
                            break;
                        }
                        prefix = app->callee();
                    }
                    if (prefix != callee)
                        throw std::invalid_argument("recipe normalization changed the verified application prefix");
                    value = world().app(value, args[i]);
                }
                break;
            }
            case RewritePlan::Kind::Tuple: value = world().tuple(args); break;
            case RewritePlan::Kind::Extract: {
                auto arity = Lit::isa(args.front()->arity());
                if (!arity || entry.bits >= *arity)
                    throw std::invalid_argument("recipe projection has unknown or invalid arity");
                value = world().extract(args.front(), *arity, entry.bits);
                break;
            }
            case RewritePlan::Kind::Instantiate: {
                RHSImporter rhs(graph_, world(), budget_);
                for (auto [old, next] : bindings_)
                    rhs.map(old, next);
                const Rule* rule = nullptr;
                for (auto [meta, source] : entry.bindings) {
                    if (meta->isa<Lam>()) continue;
                    auto var = meta->isa<Var>();
                    if (!var)
                        if (auto ex = meta->isa<Extract>()) var = ex->tuple()->isa<Var>();
                    if (var) rule = var->binder()->isa<Rule>();
                    if (rule) break;
                }
                std::unique_ptr<RulePattern> pattern;
                if (rule && rule->is_set()) {
                    pattern      = std::make_unique<RulePattern>(rule);
                    rhs.pattern  = pattern.get();
                    rhs.captures = &entry.bindings;
                }
                for (auto [meta, source] : entry.bindings) {
                    if (auto var = meta->isa<Var>(); var && var->binder()->isa<Lam>()) continue;
                    bool local
                        = std::ranges::any_of(entry.bindings, [&](const std::pair<const Def*, const Def*>& binding) {
                              auto lam = binding.first->isa<Lam>() ? binding.second->isa<Lam>() : nullptr;
                              return lam && lam->has_var() && source->has_free_var(lam->has_var());
                          });
                    if (!local) rhs.map(meta, importer_.rewrite(source));
                }
                if (pattern && std::ranges::none_of(entry.bindings, [](const std::pair<const Def*, const Def*>& pair) {
                        return pair.first->isa<Lam>()
                            || (pair.first->isa<Var>() && pair.first->as<Var>()->binder()->isa<Lam>());
                    }))
                    pattern->bind_derived(rhs, entry.bindings);
                for (auto def : protected_)
                    if (auto imported = importer_.lookup(def)) retained_.insert_or_assign(def, imported);
                value = rhs.rewrite(entry.def);
                break;
            }
        }
        if (!value || &value->world() != &world()) throw std::invalid_argument("recipe produced a foreign value");
        return value;
    }

    const GraphView& graph_;
    RebuildBudget budget_;
    GraphImporter importer_;
    RewriteOptions options_;
    DefMap<std::pair<const RewritePlan*, size_t>> replacements_;
    std::unordered_map<const RewritePlan*, std::vector<const Def*>> memo_;
    DefMap<const Def*> results_;
    Lam* new_scope_      = nullptr;
    bool region_changed_ = false;
    DefSet environment_;
    std::vector<std::pair<const Var*, const Def*>> bindings_;
    std::vector<std::pair<const Lam*, const Lam*>> region_lams_;
    DefSet protected_;
    DefMap<const Def*> retained_;
};

GraphRewrite::GraphRewrite(World* source,
                           std::unique_ptr<World> target,
                           std::shared_ptr<const GraphSnapshot> snapshot,
                           std::vector<const Def*> roots,
                           Lam* scope,
                           size_t replacements,
                           bool changed)
    : source_(source)
    , target_(std::move(target))
    , snapshot_(std::move(snapshot))
    , result_snapshot_(GraphSnapshot::capture(*target_))
    , roots_(std::move(roots))
    , scope_(scope)
    , replacements_(replacements)
    , changed_(changed) {}

GraphRewrite::GraphRewrite(GraphRewrite&&)            = default;
GraphRewrite& GraphRewrite::operator=(GraphRewrite&&) = default;
GraphRewrite::~GraphRewrite()                         = default;

void GraphRewrite::check_live() const {
    if (!result_snapshot_ || !result_snapshot_->is_live()) throw std::invalid_argument("rebuilt graph result is stale");
}

World& GraphRewrite::world() const {
    check_live();
    return published_ ? *source_ : *target_;
}

void GraphRewrite::publish(const GraphView& graph) {
    check_live();
    if (published_ || !scope_ || snapshot_ != graph.snapshot() || source_ != &graph.world() || !graph.is_current())
        throw std::invalid_argument("cannot publish an unscoped, stale, or already published graph rewrite");
    swap(*source_, *target_);
    result_snapshot_->world_ = source_;
    snapshot_->published_    = true;
    published_               = true;
}

GraphRewrite rewrite_graph(const GraphView& graph, std::span<const RewritePlan* const> plans, RewriteOptions options) {
    if (!graph.is_current() || plans.empty())
        throw std::invalid_argument("graph rewrite requires a current snapshot and plans");
    auto target = graph.world().inherit();
    if (!options.max_source_nodes)
        options.max_source_nodes = 64 * (graph.world().curr_gid() + graph.nodes().size() + 1);
    GraphRebuilder rewriter(graph, plans, *target, options);
    return rewriter.build(std::move(target));
}

} // namespace mim::pattern
