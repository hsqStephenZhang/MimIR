#include "mim/pattern/rewrite.h"

#include <stdexcept>

#include <fe/restore.h>

#include "mim/rewrite.h"
#include "mim/world.h"

namespace mim::pattern {

RewritePlan::RewritePlan(const GraphMatch& match)
    : snapshot_(match.snapshot())
    , reads_(match.reads().begin(), match.reads().end()) {
    if (!snapshot_->is_live()) throw std::invalid_argument("match snapshot has already been published");
    for (auto def : match.nodes())
        matched_.insert(def);
    for (auto def : match.captures_.captures_)
        if (def) captured_.insert(def);
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

void RewritePlan::replace(const Def* old, Recipe recipe) {
    check(recipe);
    if (!old || !matched_.contains(old) || !replacements_.emplace(old, recipe.index_).second)
        throw std::invalid_argument("replacement must name a distinct matched node");
}

namespace {

struct RebuildBudget {
    RewriteOptions options;
    size_t depth = 0, work = 0;
    void touch() {
        if (++work > options.max_source_nodes) throw std::invalid_argument("graph rebuild source work budget exceeded");
        if (depth >= options.max_depth) throw std::invalid_argument("graph rebuild depth budget exceeded");
    }
};

class GraphImporter : public Rewriter {
public:
    GraphImporter(const GraphView& graph, World& target, RebuildBudget& budget)
        : Rewriter(target)
        , graph_(graph)
        , budget_(budget) {}
    const Def* rewrite(const Def* def) override {
        budget_.touch();
        auto depth = fe::Restore(budget_.depth, budget_.depth + 1);
        return Rewriter::rewrite(def);
    }
    const Def* lookup(const Def* def) override {
        if (auto mapped = Rewriter::lookup(def)) return mapped;
        // Locally constructed test/client axioms use the global plugin ID, which is not an annex.
        if (auto axm = def->isa<Axm>(); axm && axm->plugin() == Annex::Global_Plugin) return nullptr;
        if (auto flags = graph_.world().annexes().opaque(def)) return map(def, world().annex(*flags));
        return nullptr;
    }
    const Def* rewrite_imm_Axm(const Axm* axm) override {
        auto value = Rewriter::rewrite_imm_Axm(axm)->as<Axm>();
        if (axm->plugin() == Annex::Global_Plugin || !axm->is_closed()) return value;
        // Rule templates and other functions may hold separate copies of a named polymorphic
        // axiom. Its dependent Pi is not hash-consed, so share equivalent closed imports here.
        auto& copies = axioms_[axm->flags()];
        for (auto existing : copies) {
            budget_.touch();
            if (existing->normalizer() == value->normalizer() && existing->curry() == value->curry()
                && existing->trip() == value->trip() && Check::test_alpha(existing->type(), value->type()))
                return existing;
        }
        copies.push_back(value);
        return value;
    }

private:
    const GraphView& graph_;
    RebuildBudget& budget_;
    std::unordered_map<flags_t, std::vector<const Axm*>> axioms_;
};

} // namespace

class GraphRebuilder : public Rewriter {
public:
    GraphRebuilder(const GraphView& graph,
                   std::span<const RewritePlan* const> plans,
                   World& target,
                   RewriteOptions options)
        : Rewriter(target)
        , graph_(graph)
        , budget_{options}
        , importer_(graph, target, budget_) {
        if (!graph.is_current()) throw std::invalid_argument("graph snapshot is stale");
        size_t count = 0;
        for (auto plan : plans) {
            if (!plan || plan->snapshot_ != graph.snapshot() || plan->replacements_.empty())
                throw std::invalid_argument("plan does not belong to this snapshot or has no replacements");
            if (plan->entries_.size() > options.max_recipes - std::min(count, options.max_recipes))
                throw std::invalid_argument("replacement recipe budget exceeded");
            count += plan->entries_.size();
            memo_.emplace(plan, std::vector<const Def*>(plan->entries_.size(), nullptr));
            for (auto [def, recipe] : plan->replacements_)
                if (!graph.is_pure(def) || !replacements_.emplace(def, std::pair{plan, recipe}).second)
                    throw std::invalid_argument("replacement is impure, unsupported, or overlaps another plan");
            for (const auto& entry : plan->entries_) {
                if (entry.def && &entry.def->world() != &graph.world())
                    throw std::invalid_argument("recipe references another World");
                if (entry.kind == RewritePlan::Kind::Import && !graph.is_contained(entry.def)
                    && !plan->captured_.contains(entry.def))
                    throw std::invalid_argument("recipe imports a value outside the graph");
                if (entry.kind == RewritePlan::Kind::Call && !graph.has_purity(entry.def->flags()))
                    throw std::invalid_argument("recipe calls an operator without a purity guarantee");
                if (entry.kind == RewritePlan::Kind::Literal && entry.def->node() != Node::Nat && !Idx::isa(entry.def))
                    throw std::invalid_argument("literal recipes require a Nat or Idx scalar type");
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
                    if (++scanned > options.max_source_nodes)
                        throw std::invalid_argument("source dependency budget exceeded");
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
            if (depth >= options.max_depth) throw std::invalid_argument("replacement dependency depth exceeded");
            state[def] = 1;
            for (auto next : dependencies[def])
                visit(next, depth + 1);
            state[def] = 2;
        };
        for (auto [def, replacement] : replacements_)
            visit(def, 0);
        // A decoded pure operation proves its own curry spine, not function-valued inputs.
        DefSet pure;
        for (auto def : graph.operations())
            if (graph.is_pure(def))
                for (auto cur = def; auto app = cur->isa<App>(); cur = app->callee()) {
                    budget_.touch();
                    pure.insert(cur);
                }
        for (auto def : graph.nodes())
            if (def->isa<App>() && !pure.contains(def)) protected_.insert(def);
    }

    const Def* rewrite(const Def* def) override {
        budget_.touch();
        auto active      = fe::Restore(budget_.depth, budget_.depth + 1);
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
            throw std::invalid_argument("replacement does not preserve the source type");
        auto allowed = new_scope_ ? new_scope_->has_var() : nullptr;
        for (auto var : value->free_vars())
            if (var != allowed) throw std::invalid_argument("replacement escapes the selected scope");
        results_.emplace(def, value);
        return value;
    }

    const Def* rewrite_mut(Def* def) override { return map(def, importer_.rewrite(def)); }
    const Def* rewrite_imm_Axm(const Axm* def) override { return importer_.rewrite(def); }
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
                importer_.map(var, new_scope_->var());
                map(var, new_scope_->var());
            }
            auto filter = importer_.rewrite(scope->filter());
            // Both bodies see the same unset scope stub, so self-applications cannot unfold one side.
            auto original = importer_.rewrite(scope->body());
            auto body     = rewrite(scope->body());
            changed       = body != original;
            new_scope_->set(filter, body);
            if (scope->is_external()) new_scope_->externalize();
            // Other functions use a separate map: a hash-consed value shared across scopes must
            // retain its original implementation outside the selected function body.
            for (auto ext : graph_.world().externals().muts())
                if (ext != scope)
                    if (auto copy = importer_.rewrite(ext)->isa_mut(); copy && !copy->is_external())
                        copy->externalize();
        } else {
            for (auto root : graph_.roots())
                if (root->has_free_vars())
                    throw std::invalid_argument("open graph rewrite requires an explicit function scope");
        }
        auto roots = std::vector<const Def*>();
        for (auto root : graph_.roots()) {
            auto original = new_scope_ ? nullptr : importer_.rewrite(root);
            auto value    = rewrite(root);
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
            if (!reachable.insert(def).second || def->is_mutable() || def->node() == Node::Lam
                || def->node() == Node::Pi || def->node() == Node::Var || def->node() == Node::Axm)
                continue;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        for (auto def : protected_) {
            auto i = retained_.find(def);
            if (i == retained_.end() || !reachable.contains(i->second))
                throw std::invalid_argument("replacement removes a computation without a purity proof");
        }
        world().verify();
        if (!graph_.is_current()) throw std::invalid_argument("graph changed while building replacements");
        return GraphRewrite(&graph_.world(), std::move(target), graph_.snapshot(), std::move(roots), new_scope_,
                            replacements_.size(), changed);
    }

private:
    const Def* evaluate(const RewritePlan* plan, size_t index, size_t depth) {
        if (depth >= budget_.options.max_depth || budget_.depth >= budget_.options.max_depth)
            throw std::invalid_argument("recipe evaluation depth exceeded");
        auto active = fe::Restore(budget_.depth, budget_.depth + 1);
        auto& value = memo_.at(plan)[index];
        if (value) return value;
        const auto& entry = plan->entries_[index];
        auto args         = DefVec();
        for (auto arg : entry.args)
            args.push_back(evaluate(plan, arg.index_, depth + 1));
        switch (entry.kind) {
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
        }
        if (!value || &value->world() != &world()) throw std::invalid_argument("recipe produced a foreign value");
        return value;
    }

    const GraphView& graph_;
    RebuildBudget budget_;
    GraphImporter importer_;
    DefMap<std::pair<const RewritePlan*, size_t>> replacements_;
    std::unordered_map<const RewritePlan*, std::vector<const Def*>> memo_;
    DefMap<const Def*> results_;
    Lam* new_scope_ = nullptr;
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
    GraphRebuilder rewriter(graph, plans, *target, options);
    return rewriter.build(std::move(target));
}

} // namespace mim::pattern
