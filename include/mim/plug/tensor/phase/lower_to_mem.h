#pragma once

#include <functional>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/phase.h>

#include "mim/plug/tensor/phase/panel_layout.h"

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
    const Def* lower_as_vectors(const App*);
    std::pair<const Def*, const Def*> storage_access(const Def*, const Def*);
    const Def* lower_set(const App*);
    const Def* lower_splat(const App*);
    const Def* lower_generate(const App*);

    using Layout = panel::Layout;
    DefVec layout_coords(const Def*, const Def*, Defs);
    const Def* layout_lam(const Def*, const Def*);
    void plain(const Def*) const;
    bool nat_multiple(const Def*, u64) const;
    u64 vector_width(const Def*, const Def*, const Def*, const Def*, const Def*, u64) const;
    const Def* lower_broadcast(const App*);
    const Def* lower_map_reduce(const App*, const Destination* = nullptr);
    const Def* lower_pad(const App*);
    const Def* lower_concat(const App*);
    const Def* lower_gather(const App*);
    const Def* lower_scatter(const App*);

    const Def* lower_tiled(const App*, const Destination* = nullptr);
    const Def* lower_tiled_n(const App*);
    const Def* lower_tile_fold(const App*, const Destination* = nullptr);
    /// A warp's accumulator held as tensor-core fragments: every lane keeps a value of `count` `«4; F32»` of the
    /// `rows × cols` tile, fragment `mi · cols / n + ni` covering rows `mi · m + …` and columns `ni · n + …` in the PTX
    /// layout.
    struct Fragments {
        u64 rows, cols, m, n;
        u64 count() const { return rows / m * (cols / n); }
    };
    const Def* lower_mma(const App*, const Destination&, const Fragments&);
    const Def* store_fragments(const Def* source, const Fragments&, const Destination&);
    DefVec staged_panels(const Def* root, const Def* var) const;
    const Def* merge_of(const Def*) const;
    std::tuple<const Def*, u64, const Def*, u64> peel_step(const Def*) const;
    const Def* lower_bound_fold(const App*,
                                Lam*,
                                const Def*,
                                const Def*,
                                const Def*,
                                u64,
                                const Def*,
                                const Def*,
                                const Def*,
                                const Def*,
                                bool,
                                bool,
                                const Destination* = nullptr);
    const Def* lower_product_fold(const App*);
    const Def* lower_stage(const App*);
    const Def* lower_prepack(const App*);
    std::pair<DefVec, DefVec> capture_producers(const Def*);
    bool linear_step(const Def*, const Def*, const Def*);
    const Def* lower_into(const Def*, const Destination&);
    const Def* lower_tile_body(const GridTilePlan&, const Def*, Defs, const Def*, Defs, Scratch*, const Destination*);
    bool is_view(const Def* input);
    static size_t input_count(const Def* inputs);
    std::vector<bool> view_flags(const Def* inputs);
    const Def* view_sources(const Def* inputs, const std::vector<bool>& views);
    const Def* view_reads(const Def* inputs, const Def* proxy, const std::vector<bool>& views);
    const Def* region_input(const Def*);
    void forward_storage(const Def* inputs, const Def* proxy);
    void forward_layouts(const Def* buffers, const Def* params);
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
    /// The dim a staged panel's fill runs fastest, where the source is contiguous.
    DefMap<size_t> fill_inner_;
    /// How many elements along that dim a staged panel's fill moves per step.
    DefMap<u64> fill_vector_;
    /// The capacities and layouts `tensor.placed` assigns to stages, and to the generates that fill them.
    DefMap<const Def*> placed_layouts_, fill_layouts_;
    /// The panels stored in a non-identity layout, by buffer; every access goes through the layout term's interpreter.
    DefMap<const Def*> layouts_;
    /// Read-only layout fields used only to decide whether a panel read can stay vectorized.
    DefMap<Layout> layout_views_;
    /// The logical shapes of those panels.
    DefMap<const Def*> logical_shapes_;
    /// The work-group panels, by buffer: 16-byte aligned, so their consumers read them a vector at a time.
    DefSet panels_;
    /// Nat terms known to be multiples of the value: the tile origins of the regions being lowered.
    DefMap<u64> multiples_;
    /// Runtime-shaped buffers the current tile body allocates, as `(rank, shape, type, buffer)`, freed at its end.
    std::vector<std::array<const Def*, 4>>* transients_ = nullptr;
    bool thread_tile_                                   = false;
    void transient(const Def* rank, const Def* shape, const Def* type, const Def* buffer);
    /// Facts of the tiles being lowered, in the new world: `(end, extent)` and `(last, extent - 1)` pairs bound clamps,
    /// and `(ext, nullptr)` marks a non-empty tile extent.
    std::vector<std::pair<const Def*, const Def*>> tile_ends_;
    Scratch* scratch_ = nullptr;
    /// The binding of the innermost bound region being lowered: 0 outside, 1 work-group, 2 work-item.
    u64 bound_level_ = 0;
    /// The origin var of the work-item region being lowered; a value free of it is the same in every work-item.
    const Var* item_origin_ = nullptr;
    bool uniform_           = false;
    /// While a consumer rewrites its inputs, their uniform fills are `staged`; it then fences them once.
    bool staging_       = false;
    bool fence_pending_ = false;
    /// The fills of a pipelined fold copy asynchronously; its loop waits for them before its fence.
    bool async_fill_ = false;
    /// Whether tensor-core operands are rounded to TF32 to nearest before `mma` (`-X tensor:tf32-rna`).
    bool tf32_rna_ = false;
    /// `(mem, buffer)` the next uniform fill writes instead of allocating; it leaves its memory in filled_mem_.
    std::optional<std::pair<const Def*, const Def*>> fill_target_;
    /// While a pipelined fold sizes its ring: a uniform fill only allocates its panel.
    bool probe_fill_       = false;
    const Def* filled_mem_ = nullptr;
    /// The work-group buffers that uniform fills wrote.
    DefSet shared_fills_;
    /// The warp accumulators in fragment form, by value.
    DefMap<Fragments> fragments_;
    /// While a tensor-core fold stages its panels: their fills store each source's contiguous dim innermost.
    bool mma_panels_ = false;
    bool uniform(const Def*) const;
    std::function<const Def*(const Def*)> cooperative();
    const Def* shared_mem(const Def*);
};

} // namespace mim::plug::tensor::phase
