#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Merge triples of standard Nat or wrapping integer matrix products sharing their left operand. This
/// explicit phase runs before lower_tensor; width and total-width checks are conservative.
class CombineProducts : public Phase {
public:
    using Phase::Phase;

private:
    void start() override;
};

} // namespace mim::plug::tensor::phase
