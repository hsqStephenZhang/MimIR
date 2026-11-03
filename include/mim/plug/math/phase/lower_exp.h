#pragma once

#include <mim/phase.h>

namespace mim::plug::math::phase {

/// Expands the F32 `math.exp.exp` into range reduction, a polynomial and an exponent bit construction.
/// The result consists of plain arithmetic, so the LLVM loop and SLP vectorizers can vectorize it.
/// Its error is within about one ulp of `expf` for normal results (the Cephes `expf` polynomial).
/// Subnormal results are computed by a split power of two and may differ in the last bits.
class LowerExp : public RWPhase {
public:
    using RWPhase::RWPhase;

    void start() final;

private:
    const Def* rewrite_imm_App(const App*) final;
};

} // namespace mim::plug::math::phase
