#pragma once

#include "mim/pattern/graph.h"

namespace mim::pattern {

struct WitnessResult {
    CheckResult status;
    const Def* value = nullptr;
    std::string reason;
};

struct Witness {
    size_t num_inputs;
    WitnessResult (*evaluate)(const GraphView&, Defs, World&);
};

using Witnesses = ankerl::unordered_dense::map<flags_t, Witness>;

bool is_witness_data(const Def*);

} // namespace mim::pattern
