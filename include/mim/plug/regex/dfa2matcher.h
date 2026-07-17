#pragma once

#include <cstdint>

#include <vector>

#include <automaton/dfa.h>

#include "mim/plug/regex/regex.h"

/// You can dl::get this function.
extern "C" MIM_EXPORT const mim::Def* dfa2matcher(mim::World&, const automaton::DFA&, const mim::Def*);

namespace mim::plug::regex {

struct TableTransition {
    std::uint8_t lo;     ///< Inclusive unsigned lower byte bound.
    std::uint8_t hi;     ///< Inclusive unsigned upper byte bound.
    nat_t        target; ///< Static successor state id.
};

/// Host-side form of `%regex.dfa.State`, produced only from a closed table.
struct TableState {
    bool accepting; ///< Accept when NUL is observed in this state.
    nat_t fallback; ///< Successor when no explicit range matches.
    std::vector<TableTransition> transitions;
};

/// Decoded host-side form of `%regex.DFA`.
struct DFATable {
    nat_t entry; ///< Initial state id.
    nat_t error; ///< Designated immediate-rejection state id.
    std::vector<TableState> states;
};

/// Converts an automaton DFA into the compact host-side table form used by
/// `%regex.DFA`. Missing transitions fall back to the designated error state.
DFATable dfa_to_table(World&, const automaton::DFA&);

/// Materializes a host-side DFA table as a closed MimIR `%regex.DFA` value.
const Def* encode_dfa_table(World&, const DFATable&);

/// Specializes a closed DFA table into a full-string matcher.
///
/// All state continuations are allocated before any body is emitted. This
/// forms a finite specialization cache: cyclic edges refer to existing
/// placeholders rather than recursively expanding the table interpreter.
/// Transition ranges are emitted as constant unsigned byte comparisons;
/// unmatched bytes use `fallback`, and NUL selects accept/reject from the
/// current state's `accepting` flag.
///
/// The returned function has CPS type
/// `Cn [%mem.M 0, Str n, Idx n, Cn [%mem.M 0, Bool, Idx n]]`.
const Def* dfa_table2matcher(World&, const DFATable&, const Def* n);

} // namespace mim::plug::regex
