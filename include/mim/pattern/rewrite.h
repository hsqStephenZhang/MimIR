#pragma once

#include "mim/pattern/graph_match.h"
#include "mim/pattern/import.h"

namespace mim::pattern {

class RewritePlan;
class GraphRebuilder;

class InstantiationFailure : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class Recipe {
private:
    Recipe(const RewritePlan* plan, size_t index)
        : plan_(plan)
        , index_(index) {}
    const RewritePlan* plan_;
    size_t index_;
    friend class RewritePlan;
    friend class GraphRebuilder;
};

/// A typed replacement DAG attached to one validated match. No source IR is built or modified here.
/// Reusing a Recipe reuses its result, including across multiple replacement roots.
class RewritePlan {
public:
    explicit RewritePlan(const GraphMatch&);
    RewritePlan(const RewritePlan&)            = delete;
    RewritePlan& operator=(const RewritePlan&) = delete;

    Recipe import(const Def*);
    Recipe literal(const Def* type, flags_t bits);
    Recipe call(const Axm*, std::vector<Recipe> argument_groups);
    Recipe tuple(std::vector<Recipe>);
    Recipe extract(Recipe, size_t index);
    Recipe instantiate(const Def*, const Def2Def&);
    void witness_values(std::shared_ptr<World> world) { witness_world_ = std::move(world); }
    Recipe witness_value(const Def*);
    void replace(const Def* matched_node, Recipe);

private:
    enum class Kind { Import, Literal, Call, Tuple, Extract, Instantiate, Witness };
    struct Entry {
        Kind kind;
        const Def* def           = nullptr;
        flags_t bits             = 0;
        std::vector<Recipe> args = {};
        Def2Def bindings         = {};
    };
    void check(Recipe) const;
    Recipe append(Entry);
    std::shared_ptr<const GraphSnapshot> snapshot_;
    std::vector<const Def*> reads_;
    DefSet matched_;
    std::vector<Entry> entries_;
    DefMap<size_t> replacements_;
    std::shared_ptr<World> witness_world_;
    friend class GraphRebuilder;
};

/// Zero source work selects a size-relative limit; max_depth bounds recipes, not source DAGs.
struct RewriteOptions {
    size_t max_recipes      = 10'000;
    size_t max_source_nodes = 0;
    size_t max_depth        = 256;
};

/// Owns a fully checked prospective World. Dropping this object discards the proposed change.
class GraphRewrite {
public:
    GraphRewrite(GraphRewrite&&);
    GraphRewrite& operator=(GraphRewrite&&);
    ~GraphRewrite();
    World& world() const;
    std::span<const Def* const> roots() const {
        check_live();
        return roots_;
    }
    Lam* scope() const {
        check_live();
        return scope_;
    }
    size_t replacements() const { return replacements_; }
    bool is_published() const { return published_; }
    /// Compares canonical results with an unreplaced import using the same target binder mappings.
    bool is_changed() const {
        check_live();
        return changed_;
    }
    /// Atomically swaps the rebuilt World into the source, invalidating this snapshot's matches/plans.
    /// For a scoped graph all externals are preserved and only the selected function body is rewritten.
    /// An unscoped graph returns replacement roots; it cannot be published as a whole program.
    void publish(const GraphView&);

private:
    void check_live() const;
    GraphRewrite(World* source,
                 std::unique_ptr<World> target,
                 std::shared_ptr<const GraphSnapshot> snapshot,
                 std::vector<const Def*> roots,
                 Lam* scope,
                 size_t replacements,
                 bool changed);
    World* source_;
    std::unique_ptr<World> target_;
    std::shared_ptr<const GraphSnapshot> snapshot_;
    std::shared_ptr<GraphSnapshot> result_snapshot_;
    std::vector<const Def*> roots_;
    Lam* scope_;
    size_t replacements_;
    bool changed_;
    bool published_ = false;
    friend class GraphRebuilder;
};

/// Rejects stale snapshots, overlapping/read-write-conflicting plans, impure operators,
/// replacement cycles, scope escapes and type changes before exposing a prospective World.
GraphRewrite rewrite_graph(const GraphView&, std::span<const RewritePlan* const>, RewriteOptions = {});

} // namespace mim::pattern
