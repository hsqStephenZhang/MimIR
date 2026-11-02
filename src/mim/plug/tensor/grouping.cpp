#include <limits>

#include <mim/rewrite.h>

#include <mim/pattern/witness.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

namespace {

std::optional<u64> mul(std::optional<u64> a, std::optional<u64> b) {
    if (!a || !b || (*b && *a > std::numeric_limits<u64>::max() / *b)) return {};
    return *a * *b;
}

bool pure_scalar(const Def* fn) {
    auto value = Lam::isa_ret_arg(fn);
    if (!value) return false;
    DefVec pending{value};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i > 4096) return false;
        auto def = pending[i];
        if (!seen.insert(def).second || def->isa<Var>()) continue;
        if (auto app = def->isa<App>()) {
            auto axm = App::uncurry_callee(app)->isa<Axm>();
            if (!axm) return false;
            auto name = axm->sym().str();
            if (!(name.starts_with("core.wrap.") || name.starts_with("core.nat.") || name.starts_with("core.icmp.")
                  || name.starts_with("core.ncmp.") || name.starts_with("core.conv.") || name.starts_with("core.bit")
                  || name.starts_with("core.extrema.") || name == "core.select" || name.starts_with("math.arith.")
                  || name.starts_with("math.cmp.")))
                return false;
        }
        if (def->is_mutable()) return false;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return true;
}

class CostImporter : public Rewriter {
public:
    using Rewriter::Rewriter;

    const Def* rewrite(const Def* def) override {
        if (auto value = lookup(def)) return value;
        if (def->isa<Var>()) return map(def, world().bot(Rewriter::rewrite(def->type())));
        return Rewriter::rewrite(def);
    }
};

class TileCost {
public:
    TileCost(World& w)
        : world_(w)
        , importer_(w) {}

    std::optional<u64> count(const pattern::DataOccurrence& occurrence, const Def* reader, u64 whole) {
        if (occurrence.regions.empty() && !occurrence.trips) return {};
        return region(occurrence, reader, whole, 0);
    }

private:
    std::optional<u64> cells(const Def* shape, u64 rank) {
        std::optional<u64> result = 1;
        auto value                = importer_.rewrite(shape);
        for (u64 d = 0; d != rank; ++d)
            result = mul(result, Lit::isa(value->proj(rank, d)));
        return result;
    }

    std::optional<u64> region(const pattern::DataOccurrence& occurrence, const Def* reader, u64 whole, size_t i) {
        if (i == occurrence.regions.size()) {
            if (auto slice = Axm::isa<tensor::slice>(reader)) {
                auto [meta, shape, params, value] = slice->uncurry_args<4>();
                auto rank                         = Lit::isa(meta->proj(2, 1));
                if (!rank) return {};
                auto [off, step, ext] = params->projs<3>();
                step                  = importer_.rewrite(step);
                for (u64 d = 0; d != *rank; ++d)
                    if (Lit::isa(step->proj(*rank, d)) != 1) return {};
                return cells(ext, *rank);
            }
            return whole;
        }
        auto call = occurrence.regions[i];
        if (auto fold = Axm::isa<tile_fold>(call)) {
            auto [type, blocks, init, body] = fold->uncurry_args<4>();
            auto n                          = Lit::isa(importer_.rewrite(blocks->proj(2, 0)));
            return mul(n, region(occurrence, reader, whole, i + 1));
        }
        auto tiled = Axm::isa<tensor::tiled>(call);
        if (!tiled) return {};
        auto [meta, shapes, body] = tiled->uncurry_args<3>();
        auto rank                 = Lit::isa(meta->proj(2, 1));
        auto fn                   = body->isa<Lam>();
        if (!rank || *rank > 8 || !fn || !fn->is_set()) return {};
        auto [shape, tile] = shapes->projs<2>();
        DefVec extents(*rank);
        auto axis = [&](this auto&& axis, size_t d) -> std::optional<u64> {
            if (d == *rank) {
                importer_.push();
                if (auto var = fn->has_var()) {
                    importer_.bind_old(var);
                    importer_.map(
                        var, world_.tuple({world_.bot(world_.arr(*rank, world_.type_nat())), world_.tuple(extents)}));
                }
                auto result = region(occurrence, reader, whole, i + 1);
                importer_.pop();
                return result;
            }
            auto s = Lit::isa(importer_.rewrite(shape)->proj(*rank, d));
            auto t = Lit::isa(importer_.rewrite(tile)->proj(*rank, d));
            if (!s || !t || !*t) return {};
            auto k = std::min(*s, *t);
            if (!k) return 0;
            extents[d] = world_.lit_nat(k);
            auto full  = mul(*s / k, axis(d + 1));
            if (*s % k == 0) return full;
            extents[d] = world_.lit_nat(*s % k);
            auto tail  = axis(d + 1);
            if (!full || !tail || *full > std::numeric_limits<u64>::max() - *tail) return {};
            return *full + *tail;
        };
        return axis(0);
    }

    World& world_;
    CostImporter importer_;
};

} // namespace

pattern::WitnessResult fusion_budget(const pattern::GraphView& source, Defs inputs, World& target) {
    using pattern::CheckResult;
    auto producer = graph::map_reduce(inputs[0]);
    if (!producer) return {CheckResult::False, nullptr, "fusion producer is not a map_reduce_post"};
    auto budget = Lit::isa(inputs[2]);
    if (!budget || source.complete_data_uses(producer) != CheckResult::True)
        return {CheckResult::Unknown, nullptr, "fusion budget needs complete program data uses and a literal budget"};
    auto mr    = btensor::MapRedPost::of(producer->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || !pure_scalar(mr.comb) || !pure_scalar(mr.post))
        return {CheckResult::Unknown, nullptr, "fusion needs literal ranks and proven pure scalar callbacks"};
    auto scratch = target.inherit();
    Rewriter import(*scratch);
    auto shape               = import.rewrite(mr.So);
    std::optional<u64> whole = 1;
    for (u64 d = 0; d != ranks->ro; ++d)
        whole = mul(whole, Lit::isa(shape->proj(ranks->ro, d)));
    auto limit = mul(whole, budget);
    if (!whole || !limit) return {CheckResult::Unknown, nullptr, "fusion budget shape or product is not literal"};
    u64 total = 0;
    DefSet readers;
    bool found = false;
    for (auto use : source.data_uses(producer)) {
        found |= use.consumer == inputs[1];
        if (!readers.insert(use.consumer).second) continue;
        auto occurrences = source.occurrences(use.consumer);
        if (occurrences.empty()) return {CheckResult::Unknown, nullptr, "fusion reader has no occurrence index"};
        for (const auto& occurrence : occurrences) {
            TileCost cost(*scratch);
            auto value = cost.count(occurrence, use.consumer, *whole);
            if (!value || total > std::numeric_limits<u64>::max() - *value)
                return {CheckResult::Unknown, nullptr, "fusion occurrence has unknown trips, extents or overflow"};
            total += *value;
        }
    }
    if (!found) return {CheckResult::Unknown, nullptr, "fusion reader is outside the frozen data-use index"};
    return {CheckResult::True, target.lit_bool(total <= *limit), {}};
}

} // namespace mim::plug::tensor
