#pragma once

#include "mim/pattern/graph_match.h"

namespace mim::pattern {

class RewritePlan;
class GraphRebuilder;

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
    void replace(const Def* matched_node, Recipe);

private:
    enum class Kind { Import, Literal, Call, Tuple, Extract };
    struct Entry {
        Kind kind;
        const Def* def           = nullptr;
        flags_t bits             = 0;
        std::vector<Recipe> args = {};
    };
    void check(Recipe) const;
    Recipe append(Entry);
    std::shared_ptr<const GraphSnapshot> snapshot_;
    std::vector<const Def*> reads_;
    DefSet matched_;
    std::vector<Entry> entries_;
    DefMap<size_t> replacements_;
    friend class GraphRebuilder;
};

struct RewriteOptions {
    size_t max_recipes      = 10'000;
    size_t max_source_nodes = 100'000;
    size_t max_depth        = 256;
};

/// Owns a fully checked prospective World. Dropping this object discards the proposed change.
class GraphRewrite {
public:
    GraphRewrite(GraphRewrite&&);
    GraphRewrite& operator=(GraphRewrite&&);
    ~GraphRewrite();
    World& world() const;
    std::span<const Def* const> roots() const { return roots_; }
    Lam* scope() const { return scope_; }
    size_t replacements() const { return replacements_; }
    bool is_published() const { return published_; }
    /// Atomically swaps the rebuilt World into the source, invalidating this snapshot's matches/plans.
    /// For a scoped graph all externals are preserved and only the selected function body is rewritten.
    /// An unscoped graph returns replacement roots; it cannot be published as a whole program.
    void publish(const GraphView&);

private:
    GraphRewrite(World* source,
                 std::unique_ptr<World> target,
                 std::shared_ptr<const GraphSnapshot> snapshot,
                 std::vector<const Def*> roots,
                 Lam* scope,
                 size_t replacements);
    World* source_;
    std::unique_ptr<World> target_;
    std::shared_ptr<const GraphSnapshot> snapshot_;
    std::vector<const Def*> roots_;
    Lam* scope_;
    size_t replacements_;
    bool published_ = false;
    friend class GraphRebuilder;
};

/// Rejects stale snapshots, overlapping/read-write-conflicting plans, impure operators,
/// replacement cycles, scope escapes and type changes before exposing a prospective World.
GraphRewrite rewrite_graph(const GraphView&, std::span<const RewritePlan* const>, RewriteOptions = {});

} // namespace mim::pattern
