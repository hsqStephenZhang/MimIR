#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Rewrites `%tensor.conv` to the Mim-defined `%tensor.conv_img2col_impl` template.
///
/// The phase is deliberately separate from `Lower`: it models a dialect-conversion style transform where C++ owns the
/// legality/pattern decision and Mim owns the target IR template.
class ConvToImg2Col : public RWPhase {
public:
    ConvToImg2Col(World& world, flags_t annex)
        : RWPhase(world, annex) {}

private:
    const Def* rewrite_imm_App(const App*) final;

    const Def* rewrite_via_impl(const App*, const Def* impl_annex);
};

} // namespace mim::plug::tensor::phase
