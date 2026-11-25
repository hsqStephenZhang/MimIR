#include "mim/plug/tensor/tensor.h"

#include <mim/plugin.h>

#include "mim/plug/tensor/phase/fuse.h"
#include "mim/plug/tensor/phase/lower.h"
#include "mim/plug/tensor/phase/lower_map_reduce.h"
#include "mim/plug/tensor/phase/lower_to_mem.h"
#include "mim/plug/tensor/phase/reassoc.h"

using namespace mim;
using namespace mim::plug;

namespace mim::plug::tensor {
static void reg_phases(Flags2Phases& phases) {
    Phase::hook<reassoc, phase::Reassoc>(phases);
    Phase::hook<lower_tensor, phase::Lower>(phases);
    Phase::hook<lower_map_reduce, phase::LowerMapReduce>(phases);
    Phase::hook<fuse_tensor, phase::Fuse>(phases);
    Phase::hook<lower_to_mem, phase::LowerToMem>(phases);
}
} // namespace mim::plug::tensor

// clang-format off
static constexpr PluginArg known_args[] = {
    {"reassoc-max=<n>", "Longest matrix chain whose `Catalan(n − 1)` bracketings `tensor.reassoc` enumerates and, failing one that provably wins for every extent, dispatches over at run time (default `4`; below `3` switches the dispatch off)."},
    {"vec-width=<n>", "Vector lanes the target's registers hold (`1`, `4`, `8` or `16`, default `8`). It caps the block of independent accumulators a vectorized loop body jams into its registers, and `tensor.reassoc` charges each product's vector loop in units of it; `1` means no lane-sized blocking and plain multiplication counts. See `tensor.vec_width`."},
    {"pe-cpu=<n>", "Lower CPU contractions and convolutions through the compositional `SchedTree` + `run_levels` PE interpreter (`1` on by default, `0` to fall back to `dot_schedule` / `conv_schedule`)."},
    {"group-registers=<n>", "Vector registers the CPU contraction blocks may use (default: `btensor:vregs`)."},
    {"group-cache=<n>", "Target cache budget in bytes for automatic CPU contraction block selection (default `262144`)."},
    {"cpu-mr=<n>", "Override CPU micro-kernel row tile MR (`0` uses target default)."},
    {"cpu-nr=<n>", "Override CPU micro-kernel column tile NR (`0` uses target default)."},
    {"cpu-mc=<n>", "Override CPU cache block MC (`0` uses target default)."},
    {"cpu-nc=<n>", "Override CPU outer cache block NC (`0` uses target default)."},
    {"cpu-kc=<n>", "Override CPU reduction cache block KC (`0` uses target default)."},
    {"cpu-order=<n>", "CPU BLIS loop order: `0` outer N / pack B then MC / pack A (default), `1` outer M / pack A then MC-on-N / pack B."},
    {"cpu-dyn-k=<n>", "`1` gives the CPU `K` loop a run-time tail (`tensor.sched_seq`), which halves the code of a shape whose `K` leaves one at 1–3% speed; `0` peels it statically (default `1`)."},
    {"cpu-levels=<s>", "Tile with these levels instead of the BLIS tree, like an IREE `lowering_config`: one comma-separated tile size per loop (0 keeps it whole), levels separated by `/`, outermost first."},
    {"pe-generic=<n>", "Also lower a static map-reduce that is not a contraction (elementwise, reductions) through `run_levels`, with `generic_tree` or `cpu-levels` (default `1`; `0` keeps the plain loop nest)."},
    {"pe-fuse=<n>", "Stage a single-use map-reduce that produces an input of a `run_levels` contraction inside its consumer, band by band (default `1`; `0` keeps every intermediate whole)."},
    {"cpu-pack-b=<n>", "Whether the CPU tiling packs `B`: `0` decides by the number of rows and the size of `B` (default), `1` always, `2` never."},
};
// clang-format on

MIM_PLUGIN_ENTRY(tensor) {
    plugin.register_normalizers = tensor::register_normalizers;
    plugin.register_phases      = tensor::reg_phases;
    plugin.args                 = known_args;
    plugin.num_args             = std::size(known_args);
}
