#pragma once

#include <mim/util/types.h>

namespace mim::plug::tensor::phase {

/// What the tensor phases assume of the GPU they lower for.
struct GpuLimits {
    u64 warp        = 32;
    u64 group_items = 1024;
    /// Shared memory a launch gets without opting in to more.
    u64 static_shared = 48 * 1024;
    /// Warps whose partials a split reduction merges by shuffles.
    u64 merge_warps = 4;
    /// Fewer partials than this are merged by one work-item, more by a tree.
    u64 tree_partials = 8;
};

inline constexpr GpuLimits gpu_limits{};

} // namespace mim::plug::tensor::phase
