#pragma once

#include <mim/phase.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

/// Outlines each `affine.For` whose body is `gpu.bind.group` into a `gpu.launch`.
class Outline : public RWPhase {
public:
    using Super = RWPhase;
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App*) final;
    const Def* outline(const App*);
    void captures(const Def*, const Vars&, DefSet&, DefVec&);

    const Def* sync(const Def* mem);
    const Def* store_async(const App* store);

    const Def* item_id_  = nullptr;
    const Def* k_shared_ = nullptr;
    const Def* lane_     = nullptr;
    u64 n_items_         = 1;
    bool in_item_        = false;
    bool in_async_       = false;
    DefVec splits_;
    DefSet shared_;
};

} // namespace mim::plug::gpu::phase
