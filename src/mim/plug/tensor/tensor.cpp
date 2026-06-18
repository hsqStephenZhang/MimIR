#include "mim/plug/tensor/tensor.h"

#include "mim/plugin.h"

#include "mim/plug/tensor/phase/conv_to_img2col.h"
#include "mim/plug/tensor/phase/fuse.h"
#include "mim/plug/tensor/phase/lower.h"
#include "mim/plug/tensor/phase/lower_get_set.h"
#include "mim/plug/tensor/phase/lower_map_reduce.h"
#include "mim/plug/tensor/phase/lower_to_mem.h"
#include "mim/plug/tensor/phase/op_pattern.h"

using namespace mim;
using namespace mim::plug;

namespace mim::plug::tensor {
void reg_phases(Flags2Phases& phases) {
    Phase::hook<lower_tensor, phase::Lower>(phases);
    Phase::hook<conv_to_img2col, phase::ConvToImg2Col>(phases);
    Phase::hook<lower_map_reduce, phase::LowerMapReduce>(phases);
    Phase::hook<lower_get_set, phase::LowerGetSet>(phases);
    Phase::hook<fuse_tensor, phase::Fuse>(phases);
    Phase::hook<lower_to_mem, phase::LowerToMem>(phases);
    Phase::hook<analyze_op_pattern, phase::OpPatternAnalysis>(phases);
}
} // namespace mim::plug::tensor

extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"tensor", MIM_VERSION, tensor::register_normalizers, tensor::reg_phases};
}
