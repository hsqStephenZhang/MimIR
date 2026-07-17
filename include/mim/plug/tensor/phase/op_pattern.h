#pragma once

#include <string>

#include <absl/container/flat_hash_map.h>

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

enum class OpPatternKind {
    kElemWise        = 0,
    kBroadcast       = 1,
    kInjective       = 2,
    kCommReduce      = 3,
    kOutEWiseFusable = 4,
    kTuple           = 7,
    kOpaque          = 8
};

/// Classifies a lowered tensor.map_reduce using TVM's operator-pattern lattice.
OpPatternKind classify_map_reduce(const App*);

class OpPatternAnalysis : public Analysis {
public:
    OpPatternAnalysis(World& world, flags_t annex)
        : Analysis(world, annex) {}

    OpPatternKind get_kind(const Def* def) const {
        if (auto it = patterns_.find(def); it != patterns_.end()) return it->second;
        return OpPatternKind::kOpaque;
    }

private:
    const Def* rewrite_imm_App(const App* app) override;
    std::string summarize_map_reduce_aff(const App* mra);

    absl::flat_hash_map<const Def*, OpPatternKind> patterns_;
};

} // namespace mim::plug::tensor::phase
