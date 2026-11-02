#include "mim/plug/tensor/tensor.h"

#include <mim/plugin.h>

#include <mim/pattern/witness.h>

#include "mim/plug/tensor/pattern.h"
#include "mim/plug/tensor/phase/combine_products.h"
#include "mim/plug/tensor/phase/fuse.h"
#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/phase/group_rules.h"
#include "mim/plug/tensor/phase/lower.h"
#include "mim/plug/tensor/phase/lower_map_reduce.h"
#include "mim/plug/tensor/phase/lower_to_mem.h"
#include "mim/plug/tensor/phase/reassoc.h"

using namespace mim;
using namespace mim::plug;

namespace mim::plug::tensor {
pattern::WitnessResult fusion_budget(const pattern::GraphView&, Defs, World&);
pattern::WitnessResult grouping_candidate(const pattern::GraphView&, Defs, World&);
pattern::WitnessResult reverse_candidate(const pattern::GraphView&, Defs, World&);
pattern::WitnessResult group_support_witness(const pattern::GraphView&, Defs, World&);
pattern::WitnessResult group_fusion_witness(const pattern::GraphView&, Defs, World&);
pattern::WitnessResult group_reverse_witness(const pattern::GraphView&, Defs, World&);
static pattern::WitnessResult permuted_box(Defs inputs, std::span<const u64> axes, World& target) {
    DefVec offset, extent;
    for (auto axis : axes) {
        auto o     = graph::slot(inputs[1], axes.size(), axis);
        auto e     = graph::slot(inputs[2], axes.size(), axis);
        auto start = o ? Lit::isa(o) : std::nullopt;
        auto size  = e ? Lit::isa(e) : std::nullopt;
        if (!start || !size)
            return {pattern::CheckResult::Unknown, nullptr, "input_box requires literal offsets and extents"};
        offset.push_back(target.lit_nat(*start));
        extent.push_back(target.lit_nat(*size));
    }
    return {pattern::CheckResult::True, target.tuple({target.tuple(offset), target.tuple(extent)}), {}};
}

static pattern::WitnessResult input_box_witness(const pattern::GraphView&, Defs inputs, World& target) {
    auto map  = inputs[0]->isa<Lam>();
    auto rank = Lit::isa(inputs[1]->arity());
    if (!map || !map->is_set() || !map->has_var() || !rank || !*rank || *rank > 64
        || Lit::isa(inputs[2]->arity()) != rank)
        return {pattern::CheckResult::Unknown, nullptr, "input_box needs a static rank and a set permutation map"};
    fe::Vector<u64> axes;
    fe::Vector<bool> seen(*rank, false);
    for (size_t i = 0; i != *rank; ++i) {
        auto coordinate = graph::slot(map->body(), *rank, i);
        std::optional<u64> axis;
        if (*rank == 1 && coordinate == map->has_var()) axis = 0;
        if (auto extract = coordinate ? coordinate->isa<Extract>() : nullptr;
            extract && extract->tuple() == map->has_var())
            axis = Lit::isa(extract->index());
        if (!axis || *axis >= *rank || seen[*axis])
            return {pattern::CheckResult::Unknown, nullptr, "input_box supports only bijective projection maps"};
        seen[*axis] = true;
        axes.push_back(*axis);
    }
    return permuted_box(inputs, axes, target);
}

static pattern::WitnessResult transpose_box_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    auto transpose = source.is_contained(inputs[0]) ? Axm::isa<tensor::transpose>(inputs[0]) : nullptr;
    auto rank      = Lit::isa(inputs[1]->arity());
    if (!transpose || !rank || !*rank || *rank > 64 || Lit::isa(inputs[2]->arity()) != rank)
        return {pattern::CheckResult::Unknown, nullptr, "transpose_box needs a source transpose and a static rank"};
    auto [type, permutation, shape, input] = transpose->uncurry_args<4>();
    if (Lit::isa(graph::slot(type, 2, 1)) != rank)
        return {pattern::CheckResult::Unknown, nullptr, "transpose_box rank does not agree with its source"};
    fe::Vector<u64> axes;
    fe::Vector<bool> seen(*rank, false);
    for (size_t i = 0; i != *rank; ++i) {
        auto coordinate = graph::slot(permutation, *rank, i);
        auto axis       = coordinate ? Lit::isa(coordinate) : std::nullopt;
        if (!axis || *axis >= *rank || seen[*axis])
            return {pattern::CheckResult::Unknown, nullptr, "transpose_box needs a literal bijective permutation"};
        seen[*axis] = true;
        axes.push_back(*axis);
    }
    return permuted_box(inputs, axes, target);
}

static void reg_witnesses(pattern::Witnesses& witnesses) {
    witnesses.emplace(Annex::base<input_box>(), pattern::Witness{3, input_box_witness});
    witnesses.emplace(Annex::base<transpose_box>(), pattern::Witness{3, transpose_box_witness});
    witnesses.emplace(Annex::base<within_budget>(), pattern::Witness{3, fusion_budget});
    witnesses.emplace(Annex::base<group_candidate>(), pattern::Witness{2, grouping_candidate});
    witnesses.emplace(Annex::base<reverse_legal>(), pattern::Witness{1, reverse_candidate});
    witnesses.emplace(Annex::base<group_supported>(), pattern::Witness{1, group_support_witness});
    witnesses.emplace(Annex::base<group_fusion>(), pattern::Witness{2, group_fusion_witness});
    witnesses.emplace(Annex::base<group_reverse_supported>(), pattern::Witness{3, group_reverse_witness});
}

static void reg_phases(Flags2Phases& phases) {
    Phase::hook<combine_products, phase::CombineProducts>(phases);
    Phase::hook<reassoc, phase::Reassoc>(phases);
    Phase::hook<lower_tensor, phase::Lower>(phases);
    Phase::hook<lower_map_reduce, phase::LowerMapReduce>(phases);
    Phase::hook<fuse_tensor, phase::Fuse>(phases);
    Phase::hook<group_rules, phase::GroupRules>(phases);
    Phase::hook<lower_to_mem, phase::LowerToMem>(phases);
    Phase::hook<group, phase::Group>(phases);
    Phase::hook<report_groups, phase::ReportGroups>(phases);
}
} // namespace mim::plug::tensor

// clang-format off
static constexpr PluginArg known_args[] = {
    {"group=<n>", "Enable tensor.group request selection (default 1; 0 preserves the input program)."},
    {"group-cache=<bytes>", "Scratch working-set budget for tensor.group and BLIS block selection (default 32768)."},
    {"group-registers=<n>", "Register count passed to the shared BLIS selector (default 32, at least 16)."},
    {"group-budget=<n>", "Maximum producer box recomputation factor for grouping (default 1; values above 1 permit halo requests)."},
    {"strict=<n>", "Fail on unrealized group_req markers instead of warning (default 0)."},
    {"reassoc-max=<n>", "Longest matrix chain whose `Catalan(n − 1)` bracketings `tensor.reassoc` enumerates and, failing one that provably wins for every extent, dispatches over at run time (default `4`; below `3` switches the dispatch off)."},
    {"vec-width=<n>", "Vector lanes the target's registers hold (`1`, `4`, `8` or `16`, default `8`). It caps the block of independent accumulators a vectorized loop body jams into its registers, and `tensor.reassoc` charges each product's vector loop in units of it; `1` means no lane-sized blocking and plain multiplication counts. See `tensor.vec_width`."},
};
// clang-format on

MIM_PLUGIN_ENTRY(tensor) {
    plugin.register_normalizers      = tensor::register_normalizers;
    plugin.register_phases           = tensor::reg_phases;
    plugin.register_witnesses        = tensor::reg_witnesses;
    plugin.register_graph_operations = tensor::graph::register_operations;
    plugin.args                      = known_args;
    plugin.num_args                  = std::size(known_args);
}
