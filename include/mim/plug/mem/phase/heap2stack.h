#pragma once

#include "mim/plug/mem/escape_analysis.h"

namespace mim::plug::mem::phase {

/// Turns a `mem.alloc` that does not escape its function (see mem::EscapeAnalysis) into a `mem.slot`.
///
/// The slot's continuation receives the remainder of the alloc's home.
/// Only allocations of at most `-X mem:heap2stack=<n>` elements in address space 0 qualify; `0` disables this phase.
/// A slot lives until its function returns, so an alloc in a loop of its function stays on the heap unless @p loops.
/// With @p loops, as `mem.heap2stack_loops` after the last `mem.seo`, the ll backend gives each slot one static
/// entry-block stack slot that the iterations reuse. A `mem.free` of a converted alloc is dropped.
class Heap2Stack : public RWPhase {
public:
    Heap2Stack(World&, flags_t annex, bool loops = false);

private:
    bool analyze() final;
    const Def* rewrite_imm_App(const App*) final;

    /// The number of scalars in @p type; `std::nullopt` if not statically known or more than `max_`.
    std::optional<nat_t> num_elems(const Def* type) const;

    EscapeAnalysis escape_;
    const nat_t max_;
    const bool loops_;
    /// Old tail ↦ the allocs on its chain to turn into slots, in program order.
    DefMap<fe::Vector<const App*>> slots_;
    DefSet converted_;
};

/// `Heap2Stack` that also turns allocations in loops into slots.
class Heap2StackLoops : public Heap2Stack {
public:
    Heap2StackLoops(World& world, flags_t annex)
        : Heap2Stack(world, annex, true) {}
};

} // namespace mim::plug::mem::phase
