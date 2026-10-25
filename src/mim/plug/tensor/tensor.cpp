#include "mim/plug/tensor/tensor.h"

#include <mim/plugin.h>

#include "mim/plug/tensor/phase/fuse.h"
#include "mim/plug/tensor/phase/group.h"
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
    Phase::hook<group_tensor, phase::Group>(phases);
    Phase::hook<lower_to_mem, phase::LowerToMem>(phases);
}
} // namespace mim::plug::tensor

// clang-format off
static constexpr PluginArg known_args[] = {
    {"reassoc-max=<n>", "Longest matrix chain whose `Catalan(n − 1)` bracketings `tensor.reassoc` enumerates and, failing one that provably wins for every extent, dispatches over at run time (default `4`; below `3` switches the dispatch off)."},
    {"vec-width=<n>", "Vector lanes the target's registers hold (`1`, `4`, `8` or `16`, default `8`). It caps the block of independent accumulators a vectorized loop body jams into its registers, and `tensor.reassoc` charges each product's vector loop in units of it; `1` means no lane-sized blocking and plain multiplication counts. See `tensor.vec_width`."},
    {"vec-bytes=<n>", "Bytes in one vector register of the target, for `tensor.gemm_tiles` (default: the host's)."},
    {"vec-regs=<n>", "Vector registers of the target, for `tensor.gemm_tiles` (default: the host's)."},
    {"lane-fma=<0|1>", "Whether the target's vector FMA reads its broadcast operand off a lane, as on AArch64, for `tensor.gemm_tiles` (default: the host's)."},
    {"l1=<bytes>", "L1 data cache of the target, for `tensor.gemm_tiles` (default: the host's)."},
    {"bufs=<n>", "Operand sets a GEMM's register block keeps in flight, 2 on an in-order target, for `tensor.gemm_tiles` (default: 1)."},
    {"gemm-mr=<n>", "Rows of a GEMM's register block, overriding `tensor.gemm_tiles`."},
    {"gemm-nr=<n>", "Columns of a GEMM's register block, overriding `tensor.gemm_tiles`."},
    {"gemm-kc=<n>", "K block of a GEMM, overriding `tensor.gemm_tiles`."},
    {"gemm-nc=<n>", "Columns of a GEMM's cached B block, overriding `tensor.gemm_tiles`."},
};
// clang-format on

MIM_PLUGIN_ENTRY(tensor) {
    plugin.register_normalizers = tensor::register_normalizers;
    plugin.register_phases      = tensor::reg_phases;
    plugin.args                 = known_args;
    plugin.num_args             = std::size(known_args);
}
