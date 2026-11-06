#pragma once

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/phase.h>

namespace mim::plug::tensor {
struct GridTilePlan;
}

namespace mim::plug::tensor::phase {

/// Bufferizes nominal Tensor types and operations onto the shared buffer layer.
/// The following mem.add_mem phase threads the emitted memory operands.
class LowerToMem : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    struct Destination {
        const Def *mem, *buffer, *shape, *origin, *extent;
    };
    struct Scratch {
        const Def *mem, *result_type;
        fe::Vector<std::pair<Lam*, const Def*>> bindings;
    };
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
    const Def* lower_get(const App*);
    const Def* lower_set(const App*);
    const Def* lower_splat(const App*);
    const Def* lower_generate(const App*);
    const Def* lower_broadcast(const App*);
    const Def* lower_map_reduce(const App*, const Destination* = nullptr);
    const Def* lower_pad(const App*);
    const Def* lower_concat(const App*);
    const Def* lower_gather(const App*);
    const Def* lower_scatter(const App*);

    const Def* lower_tiled(const App*, const Destination* = nullptr);
    const Def* lower_tiled_n(const App*);
    const Def* lower_tile_fold(const App*);
    const Def* lower_product_fold(const App*);
    const Def* lower_stage(const App*);
    const Def* lower_prepack(const App*);
    std::pair<DefVec, DefVec> capture_producers(const Def*);
    bool linear_step(const Def*, const Def*, const Def*);
    const Def* lower_into(const Def*, const Destination&);
    const Def* lower_tile_body(const GridTilePlan&, const Def*, Defs, const Def*, Defs, Scratch*, const Destination*);
    const Def* region_input(const Def*);
    void forward_storage(const Def* inputs, const Def* proxy);
    std::pair<const Def*, const Def*> allocate(const Def*, const Def*, const Def*, bool = true);
    void prepare_realizations(const Def*, const Lam* = nullptr, Defs = {});

    const Def* materialize(const Def*, const Def*);
    const Def* to_buffer(const Def*, const Def*);
    const Def* make_buffer(const Def*, const Def*, const Def*);
    const Def* splat_buffer(const Def*, const Def*);

    /// A `⊥: mem.M 0` placeholder consumed by emitted buffer operations; AddMem replaces it with the
    /// scheduler-placed current memory.
    /// Shared by every op whose result is a pure function of its value operands, so that genuinely equal
    /// ops still collapse into one (e.g. a weight literal materialized at many sites).
    const Def* bot_mem();

    /// AddMem resolves this request at each operand use, after threading preceding memory effects.
    const Def* fresh_mem();

    /// AddMem inlines these request continuations without fixing their memory operands at the lam head.
    const Def* wrap_fresh_mem(const Def* body);

    flags_t tensor_key_ = 0;
    bool tensors_seen_  = false;
    /// The fresh-memory continuations minted while the current lam's body is being rewritten.
    fe::Vector<Lam*> pending_;
    /// Per-lam memo for the ops that consume a fresh memory (see LowerToMem::rewrite).
    DefMap<const Def*> fresh_memo_;
    DefMap<const Def*> realizations_;
    DefMap<const Def*> storage_shapes_;
    /// Runtime-shaped buffers the current tile body allocates, as `(rank, shape, type, buffer)`, freed at its end.
    std::vector<std::array<const Def*, 4>>* transients_ = nullptr;
    void transient(const Def* rank, const Def* shape, const Def* type, const Def* buffer);
    /// Facts of the tiles being lowered, in the new world: `(end, extent)` and `(last, extent - 1)` pairs bound clamps,
    /// and `(ext, nullptr)` marks a non-empty tile extent.
    std::vector<std::pair<const Def*, const Def*>> tile_ends_;
    Scratch* scratch_ = nullptr;
};

} // namespace mim::plug::tensor::phase
