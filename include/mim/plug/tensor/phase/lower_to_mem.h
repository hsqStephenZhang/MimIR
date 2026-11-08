#pragma once

#include <functional>

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
    const Def* lower_as_vectors(const App*);
    std::pair<const Def*, const Def*> storage_access(const Def*, const Def*);
    const Def* lower_set(const App*);
    const Def* lower_splat(const App*);
    const Def* lower_generate(const App*);

public:
    /// A rotation swizzle: 16-byte chunk `c` of row `r` is stored at chunk `(c + (r >> shift) % groups) % chunks`.
    struct Swizzle {
        u64 width, groups, shift, chunks;
    };
    /// A panel's storage layout: its last dim split into `(extent / split, split)`, those dims permuted (`perm[p]` is
    /// the dim at physical position `p`), and the rows of the result swizzled.
    struct Layout {
        std::vector<size_t> perm;
        u64 split = 1;
        std::optional<Swizzle> swizzle;
        /// The dims after the split and permutation, before merging and padding.
        std::vector<u64> permuted;
        bool identity() const;
    };
    /// How the work-items of the current item level read a panel, per panel dim: the loop dim whose tile index
    /// selects it, or whether an item reads the dim whole; the remaining dims are fixed per step.
    struct PanelReads {
        std::vector<u64> counts, steps;
        std::vector<std::optional<size_t>> loop_dim;
        std::vector<bool> whole;
    };
    struct ItemGrid {
        std::vector<u64> counts, steps;
    };

private:
    std::tuple<const Def*, Layout, bool> panel_layout(const Def*, size_t, u64, u64, const PanelReads*);
    DefVec layout_coords(const Layout&, const Def*, Defs);
    const Def* layout_lam(const Layout&, const Def*);
    void plain(const Def*) const;
    std::optional<std::pair<size_t, u64>> contiguous_dim(const Def*, const Def*, const Def*);
    std::optional<size_t> origin_dim(const Def*) const;
    std::optional<PanelReads> panel_reads(const App*) const;
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
    const Def* lower_tile_fold(const App*);
    const Def* merge_of(const Def*) const;
    std::tuple<const Def*, u64, const Def*> peel_step(const Def*) const;
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
                                bool);
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
    /// The panels stored in a non-identity layout, by buffer; every access goes through LowerToMem::layout_coords or
    /// layout_map.
    DefMap<Layout> layouts_;
    /// The logical shapes of those panels.
    DefMap<const Def*> logical_shapes_;
    /// The item level's reads of a staged panel, by its generate.
    DefMap<PanelReads> panel_reads_;
    /// The work-group panels, by buffer: 16-byte aligned, so their consumers read them a vector at a time.
    DefSet panels_;
    /// The tile grid of the item level being lowered, over the tiled op's loop dims.
    std::optional<ItemGrid> item_grid_;
    /// The components of the work-group region's origin, by loop dim.
    DefMap<size_t> origin_dims_;
    /// Nat terms known to be multiples of the value: the tile origins of the regions being lowered.
    DefMap<u64> multiples_;
    /// Runtime-shaped buffers the current tile body allocates, as `(rank, shape, type, buffer)`, freed at its end.
    std::vector<std::array<const Def*, 4>>* transients_ = nullptr;
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
    bool uniform(const Def*) const;
    std::function<const Def*(const Def*)> cooperative();
    const Def* shared_mem(const Def*);
};

} // namespace mim::plug::tensor::phase
