#pragma once

#include <array>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Bufferizes nominal Tensor types and operations onto the shared buffer layer.
/// The following mem.add_mem phase threads the emitted memory operands.
class LowerToMem : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    void start() override;
    void validate();
    bool is_tensor_type(const Def*) const;
    bool mentions_tensor(const Def*) const;
    bool is_tensor_fn(Lam*) const;

    const Def* rewrite(const Def*) override;
    const Def* rewrite_mut_Lam(Lam*) override;
    const Def* rewrite_imm_Nom(const Nom*) override;
    const Def* rewrite_imm_Name(const Name*) override;
    const Def* rewrite_imm_Struc(const Struc*) override;
    const Def* rewrite_imm_App(const App*) override;
    const Def* rewrite_imm_Extract(const Extract*) override;
    const Def* buf_of(const Def*);

    const Def* lower_from_array(const App*);
    const Def* uncurry(Lam*);
    const Def* lower_get(const App*);
    const Def* lower_set(const App*);
    const Def* lower_splat(const App*);
    const Def* lower_generate(const App*);
    const Def* lower_broadcast(const App*);
    const Def* lower_map_reduce(const App*);
    const Def* lower_pe_cpu(const App*);
    /// The slots of the `run_levels_with` call for the old contraction @p app: `(xs, Sis, prods, fuse)`, where a
    /// single-use contraction producing one input becomes a `tensor.fill_of` producer instead of a buffer.
    std::array<const Def*, 4> pe_slots(const App* app,
                                       const Def* Rn,
                                       const Def* Ris,
                                       const Def* Sis,
                                       const Def* Tis,
                                       const Def* maps,
                                       const Def* a,
                                       bool may_fuse);
    /// The levels of `-X tensor:cpu-levels=<sizes>/<sizes>/…` for a contraction over @p rn loops, or `nullptr`.
    const Def* cpu_levels(u64 rn);
    /// The `tensor.fill_of` producer for the old input @p input of a slot read through @p map, or `nullptr`.
    const Def* fuse_producer(const Def* input, const Def* map, const Def* Rn, const Def* r);
    const Def* lower_pad(const App*);
    const Def* lower_concat(const App*);
    const Def* lower_gather(const App*);
    const Def* lower_scatter(const App*);
    const Def* splat_buffer(const Def*, const Def*);

    /// A `⊥: mem.M 0` placeholder consumed by emitted buffer operations; AddMem replaces it with the
    /// scheduler-placed current memory.
    /// Shared by every op whose result is a pure function of its value operands, so that genuinely equal
    /// ops still collapse into one (e.g. a weight literal materialized at many sites).
    const Def* bot_mem();

    /// A *fresh* `mem.M 0` for one emitted buffer/btensor operation: the var of a newly minted continuation
    /// `con fresh_mem(mem: mem.M 0)` that receives its memory once LowerToMem::wrap_fresh_mem has chained it
    /// in front of the enclosing lam's body.
    /// Required by every op that allocates a buffer it then writes into, for two independent reasons:
    /// 1. Immutable Def%s are hash-consed, so two `buffer.alloc`s agreeing on `(r, s, T)` and sharing one
    ///    placeholder collapse into a single allocation - two distinct tensors would alias one buffer.
    /// 2. AddMem is a memoizing Rewriter, so two operations sharing one argument tuple
    ///    `(⊥: mem.M 0, …)` - which happens whenever they differ only in their curried callee - are threaded
    ///    from the *same* current memory, and the resulting parallel mem chains collapse in `cps.conv`.
    /// Mutables are never hash-consed, so the continuations' vars are distinct by construction - no
    /// distinguishing tag required.
    const Def* fresh_mem();

    /// Chains the LowerToMem::pending_ continuations in front of the freshly rewritten @p body:
    /// yields `mem.fresh (0, k₁)` with `k₁ ↦ mem.fresh (0, k₂)`, …, and the last one carries @p body.
    /// AddMem resolves each request by jumping to the continuation with the scheduler-placed current memory,
    /// and the `tt` filter beta-reduces the continuations away again as soon as that happens.
    const Def* wrap_fresh_mem(const Def* body);

    flags_t tensor_key_ = 0;
    LamSet shape_poly_;
    bool tensors_seen_ = false;
    /// The fresh-memory continuations minted while the current lam's body is being rewritten.
    fe::Vector<Lam*> pending_;
    /// Per-lam memo for the ops that consume a fresh memory (see LowerToMem::rewrite).
    DefMap<const Def*> fresh_memo_;
    /// How many old Def%s use each old Def; a producer used once can be fused into its consumer.
    DefMap<size_t> uses_;
};

} // namespace mim::plug::tensor::phase
