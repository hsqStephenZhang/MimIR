#pragma once

#include <optional>

#include "mim/pattern/pattern.h"

namespace mim::pattern {

enum class MatchStatus { Matched, No_Match, Unknown, Budget_Exceeded };

struct MatchOptions {
    size_t max_steps = 100'000;
    size_t max_depth = 256;
};

class MatchResult {
public:
    /// Returns nullptr for an unbound capture; handles from another program are rejected.
    const Def* isa_capture(Pattern) const;

private:
    MatchResult(const PatternProgram* program, std::vector<const Def*> captures)
        : program_(program)
        , captures_(std::move(captures)) {}

    const PatternProgram* program_;
    std::vector<const Def*> captures_;
    friend class MatchEngine;
};

struct MatchOutcome {
    MatchStatus status;
    std::optional<MatchResult> result;
    size_t steps;
};

/// Read-only, single-root matching. Invalid handles throw std::invalid_argument.
/// No_Match, Unknown and exhausted budgets never expose partial bindings.
MatchOutcome match(const PatternProgram&, Pattern, const Def*, MatchOptions = {});

} // namespace mim::pattern
