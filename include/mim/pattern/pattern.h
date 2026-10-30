#pragma once

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "mim/def.h"

namespace mim::pattern {

class PatternProgram;
class MatchEngine;

/// A handle into a non-moving PatternProgram. Only explicit captures bind values.
class Pattern {
public:
    bool operator==(const Pattern&) const = default;

private:
    Pattern(const PatternProgram* program, size_t index)
        : program_(program)
        , index_(index) {}

    const PatternProgram* program_;
    size_t index_;

    friend class PatternProgram;
    friend class MatchEngine;
    friend class MatchResult;
    friend class GraphPattern;
};

enum class CheckResult { True, False, Unknown };
using Predicate = std::function<CheckResult(std::span<const Def* const>)>;

/// Append-only pattern arena. Operands must already belong to this arena, so patterns cannot cycle.
/// Building patterns never constructs or normalizes program IR.
class PatternProgram {
public:
    PatternProgram()                                 = default;
    PatternProgram(const PatternProgram&)            = delete;
    PatternProgram(PatternProgram&&)                 = delete;
    PatternProgram& operator=(const PatternProgram&) = delete;
    PatternProgram& operator=(PatternProgram&&)      = delete;

    Pattern any();
    Pattern exact(const Def*);
    Pattern literal(const Def* type, flags_t bits);
    /// Match exactly these raw operands, optionally checking the node's semantic flags.
    /// Mutable nodes are opaque: use exact() to match one by identity.
    Pattern node(Node, std::vector<Pattern> operands, std::optional<flags_t> flags = {});
    Pattern capture(Pattern);
    Pattern capture() { return capture(any()); }
    Pattern all(Pattern, Pattern);
    Pattern either(Pattern, Pattern);
    /// Existential probe in an isolated environment; captures never escape the negation.
    Pattern negate(Pattern);
    /// Deferred until structural matching has bound its captures.
    Pattern type(Pattern);
    /// Deferred until type matching finishes. Missing inputs produce Unknown.
    /// Predicates must be read-only and must not modify this program during matching.
    Pattern where(Pattern, std::vector<Pattern> captures, Predicate);
    /// Exact/literal references are tied to their source IR storage and need rebuilding after a World swap.
    bool is_current() const;

private:
    enum class Kind { Any, Exact, Literal, Node, Capture, All, Either, Not, Type, Where };

    struct Entry {
        Kind kind;
        std::vector<Pattern> operands = {};
        const Def* def                = nullptr;
        Node node                     = Node::Lit;
        std::optional<flags_t> flags  = {};
        std::vector<Pattern> inputs   = {};
        Predicate predicate           = {};
    };

    void check(Pattern) const;
    void remember(const Def*);
    Pattern append(Entry);
    std::vector<Entry> entries_;
    struct Reference {
        World* world;
        std::shared_ptr<const void> generation;
    };
    std::vector<Reference> references_;

    friend class MatchEngine;
    friend class GraphPattern;
};

} // namespace mim::pattern
