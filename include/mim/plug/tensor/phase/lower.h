#pragma once

#include <fe/restore.h>

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Lowers the high-level tensor axioms into the low-level tensor axioms (`map_reduce`, …).
/// Each high-level axiom comes with a matching `*_impl` annex (a
/// `lam` with the same signature as the axiom); the lowering simply re-applies the args
/// to the `_impl` annex. Each `_impl` body references the `_impl` variants of its
/// dependencies, so the chain of beta-reductions bottoms out at the low-level axioms in
/// one step. The resulting low-level axioms are then lowered to primitives by
/// `LowerMapReduce`.
class Lower : public RWPhase {
public:
    Lower(World& world, flags_t annex)
        : RWPhase(world, annex) {}

private:
    void start() final;
    /// Schedule rules are decided only in the program: an annex is library code, instantiated later.
    void rewrite_annex(flags_t, Sym, const Def*) final;
    const Def* rewrite_imm_App(const App*) final;
    const Def* rewrite_imm_Extract(const Extract*) final;

    /// Reads straight through a `broadcast` / `repeat` at @p index instead of materializing it.
    /// Per axis the shape op either passes the index through, reads a size-1 input axis at 0, or (for `repeat`)
    /// wraps it; an axis that is none of these decidably keeps the shape op.
    const Def* read_through(const Def* base, Defs index);

    const Def* lower_via_impl(const App*, const Def* impl_annex);
    template<annex_without_subs Id>
    const Def* impl() {
        auto _ = fe::Restore(in_annex_, true);
        return annex<Id>();
    }

    bool in_annex_ = false;
    /// `tensor.fastest_axis` applied to the dot family's right operand (of the given rank):
    /// the reflection the dot `_impl`s take as their leading argument, pre-applied here because the
    /// operand is only concrete at this staging point. The schedule decision built on the answer
    /// stays in the `_impl`'s IR (see tensor.dot_product_impl).
    const Def* fastest_axis_2(const App*, const Def* rank);
    /// Whether a dot family op produces or consumes a `tensor.compute_at` operand: it then takes the plain schedule,
    /// which neither packs its output nor blocks the loops of a staged operand's free dims inward.
    const Def* plain(const App*);

    /// The old-world operands annotated with `tensor.compute_at`.
    DefSet staged_;
};

} // namespace mim::plug::tensor::phase
