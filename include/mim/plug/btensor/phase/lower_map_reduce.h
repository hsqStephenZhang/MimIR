#pragma once

#include <functional>
#include <vector>

#include <mim/def.h>
#include <mim/phase.h>

#include <mim/util/types.h>

namespace mim::plug::btensor::phase {

struct MrOp;

/// Lowers the buffer-world operations (`btensor.map_reduce_post`, `btensor.broadcast`, `btensor.pad`,
/// `btensor.concat`, `btensor.gather`, `btensor.scatter`) into `affine.For` loop nests over
/// `buffer.read` / `buffer.write` / `buffer.alloc`, threading `mem.M`.
/// These are the buffer-world counterparts of the corresponding `tensor.*` ops; the `tensor` plugin's
/// bufferization (`tensor.lower_to_mem`) maps the SSA tensor ops onto them.
/// Also lowers `buffer.lit` into a fill loop, so a large constant/splat tensor becomes a loop rather
/// than a monolithic `mem.store` of a giant literal array (which the LLVM backend cannot digest).
class LowerMapReduce : public RWPhase {
public:
    LowerMapReduce(World& world, flags_t annex)
        : RWPhase(world, annex) {}

    /// A coordinate as `cst + Σ coef[d] · o#d` over a loop vector `o`.
    struct Lin {
        std::vector<s64> coef;
        s64 cst;
    };

private:
    /// A `compute_at` operand whose producer runs inside the consumer: once per value of the consumer's leading
    /// `level` parallel dims, over a tile of extents `ext` whose origin per producer axis is the part of `lin` (the
    /// consumer's read of that axis) over those dims; the producer writes axis a from its parallel loop `perm[a]`.
    struct Stage {
        const App* producer;
        u64 level;
        std::vector<Lin> lin;
        std::vector<u64> ext;
        std::vector<u64> perm;
        bool full = false; // level-zero producer over its whole output, including non-linear consumer maps
    };
    /// Where a staged producer writes: the tile of (unfolded) shape `ext`, axis a from parallel loop `perm[a]`.
    struct Tile {
        std::vector<u64> perm;
        const Def* ext;
    };

    void start() override;
    const Def* rewrite_imm_App(const App*) override;
    const Def* build_nest(const MrOp&,
                          const Def* Sr_loop,
                          const Def* U,
                          const DefVec& in_bufs,
                          const DefVec& in_maps,
                          const DefVec& in_shapes,
                          const Def* post_bufs,
                          const DefVec& shift,
                          const Tile*,
                          const Def* sl,
                          std::function<Lam*(const Def*)> stage);
    const Def* lower_map_reduce_post(const App*);
    const Def* lower_broadcast(const App*);
    const Def* lower_pad(const App*);
    const Def* lower_concat(const App*);
    const Def* lower_buffer_lit(const App*);
    const Def* lower_gather(const App*);
    const Def* lower_scatter(const App*);

    /// Old-world producers computed inside their consumer, and the old-world `compute_at` operands they feed.
    DefSet dropped_;
    DefMap<Stage> stages_;
    DefMap<const App*> consumers_;
};

} // namespace mim::plug::btensor::phase
