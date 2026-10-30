#pragma once

#include <string>

#include "mim/rule.h"

#include "mim/pattern/match.h"

namespace mim {
class Rewriter;
}

namespace mim::pattern {

struct RuleMatch {
    MatchStatus status;
    /// Original Rule metavariable (Var or Extract) to matched source Def. Empty on failure.
    Def2Def bindings;
    size_t steps;
};

struct RuleReplacement {
    MatchStatus status;
    const Def* value;
    size_t steps;
    std::string reason;
};

/// Compiles a Rule's LHS to the read-only matcher. Guard and RHS are not evaluated here.
/// Supports direct Rule variables and literal-index projections, with deferred structural type checks.
/// Other variables and mutable nodes match by identity. No inverse normalization or Hole filling.
/// The Rule and its World must remain unchanged and outlive this object.
class RulePattern {
public:
    explicit RulePattern(const Rule*);

    RuleMatch match_lhs(const Def*, MatchOptions = {}) const;
    /// Match a closed candidate, import captures, evaluate the guard, and build a type-preserving RHS
    /// in a separate World. Open terms need the overload taking a contextual Rewriter.
    /// Matched means a replacement was built; this method does not publish or modify any program root.
    /// Missing captures are rejected, never filled with Holes. Evaluation uses the World's PE fuel.
    RuleReplacement build_replacement(const Def*, World& target, MatchOptions = {}) const;
    /// Reuse an enclosing graph rebuild's variable/binder mappings for open candidates.
    /// Foreign free variables in the guard/RHS must also occur in the candidate.
    RuleReplacement build_replacement(const Def*, Rewriter& context, MatchOptions = {}) const;

private:
    Pattern lower(const Def*);
    bool is_meta(const Def*) const;
    bool is_bound(Defs, const Def2Def&) const;

    const Rule* rule_;
    PatternProgram program_;
    std::optional<Pattern> root_;
    DefMap<Pattern> lowered_;
    DefMap<Pattern> captures_;
};

} // namespace mim::pattern
