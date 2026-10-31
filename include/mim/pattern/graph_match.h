#pragma once

#include "mim/pattern/graph.h"
#include "mim/pattern/match.h"

namespace mim::pattern {

class GraphPattern;

class Anchor {
private:
    Anchor(const GraphPattern* graph, size_t index)
        : graph_(graph)
        , index_(index) {}
    const GraphPattern* graph_;
    size_t index_;
    friend class GraphPattern;
    friend class GraphMatch;
};

enum class AnchorDomain { Operations, Nodes };

struct UseSite {
    std::optional<size_t> group                    = {};
    std::optional<std::vector<size_t>> path        = {};
    std::optional<std::vector<size_t>> result_path = {};
};

/// Anchors share one binding environment. Multi-anchor patterns must be connected by a shared
/// positive capture or a declared data-use relation. Handles require this non-moving object to live.
class GraphPattern {
public:
    explicit GraphPattern(const PatternProgram& program)
        : program_(program) {}
    GraphPattern(const GraphPattern&)            = delete;
    GraphPattern& operator=(const GraphPattern&) = delete;

    Anchor anchor(Pattern, AnchorDomain = AnchorDomain::Operations);
    void used_by(Anchor producer, Anchor consumer, UseSite = {});
    void only_used_by(Anchor producer, Anchor consumer, UseSite = {});
    void allow_aliases(bool allow = true) { distinct_ = !allow; }

private:
    struct Entry {
        Pattern pattern;
        AnchorDomain domain;
    };
    struct Relation {
        size_t producer;
        size_t consumer;
        bool exclusive;
        UseSite site;
    };
    void check(Anchor) const;
    const PatternProgram& program_;
    std::vector<Entry> anchors_;
    std::vector<Relation> relations_;
    bool distinct_ = true;
    friend class MatchEngine;
};

class GraphMatch {
public:
    const Def* at(Anchor) const;
    const Def* isa_capture(Pattern) const;
    std::span<const Def* const> anchors() const {
        check_live();
        return anchors_;
    }
    std::span<const Def* const> reads() const {
        check_live();
        return *reads_;
    }
    /// Anchor and capture values; input captures may denote boundaries, not internal operations.
    std::span<const Def* const> nodes() const {
        check_live();
        return matched_;
    }
    std::shared_ptr<const GraphSnapshot> snapshot() const { return snapshot_; }

private:
    void check_live() const;
    GraphMatch(const GraphPattern* pattern,
               const GraphView& graph,
               std::vector<const Def*> anchors,
               MatchResult captures,
               std::shared_ptr<const std::vector<const Def*>> reads,
               std::vector<const Def*> matched)
        : pattern_(pattern)
        , snapshot_(graph.snapshot())
        , anchors_(std::move(anchors))
        , captures_(std::move(captures))
        , reads_(std::move(reads))
        , matched_(std::move(matched)) {}
    const GraphPattern* pattern_;
    std::shared_ptr<const GraphSnapshot> snapshot_;
    std::vector<const Def*> anchors_;
    MatchResult captures_;
    std::shared_ptr<const std::vector<const Def*>> reads_;
    std::vector<const Def*> matched_;
    friend class MatchEngine;
};

enum class SearchStatus { Complete, Unknown, Budget_Exceeded, Result_Limit };
enum class GraphSearchMode { Indexed, Exhaustive };

struct GraphMatchOptions {
    MatchOptions match      = {};
    size_t max_candidates   = 100'000;
    size_t max_results      = 1000;
    bool reverse_candidates = false; // Also useful for checking enumeration-order independence.
    GraphSearchMode mode    = GraphSearchMode::Indexed;
};

struct GraphMatches {
    std::vector<GraphMatch> matches;
    SearchStatus status;
    size_t steps;
    size_t candidates;
    size_t prefilter_checks = 0;
    size_t index_steps      = 0;
};

/// Complete means all anchor assignments were considered, including when matches is empty.
/// Other statuses explicitly mark incomplete enumeration; every returned match is individually valid.
GraphMatches find_matches(const GraphPattern&, const GraphView&, GraphMatchOptions = {});

} // namespace mim::pattern
