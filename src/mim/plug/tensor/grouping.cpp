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

} // namespace

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
                  || name.starts_with("math.cmp.") || name.starts_with("math.extrema.") || name.starts_with("math.exp.")
                  || name.starts_with("math.rt.") || name.starts_with("math.conv.")))
                return false;
        }
        if (def->is_mutable()) return false;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return true;
}

namespace {

class CostImporter : public Rewriter {
public:
    using Rewriter::Rewriter;

    const Def* rewrite(const Def* def) override {
        if (auto value = lookup(def)) return value;
        if (auto var = def->isa<Var>(); var && !lookup(var->binder()))
            return map(def, world().bot(Rewriter::rewrite(def->type())));
        return Rewriter::rewrite(def);
    }
};

class TileCost {
public:
    TileCost(World& w, bool enumerate = false)
        : world_(w)
        , importer_(w)
        , enumerate_(enumerate) {}

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
            if (auto slice = Axm::isa<slice_box>(reader)) {
                auto [meta, shape, params, value] = slice->uncurry_args<4>();
                auto rank                         = Lit::isa(meta->proj(2, 1));
                return rank ? cells(params->proj(3, 1), *rank) : std::nullopt;
            }
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
        if (auto tiled = Axm::isa<tiled_n>(call)) {
            Rewriter geometry(world_);
            auto plan = multi_tile_plan(geometry.rewrite(tiled)->as<App>());
            auto body = tiled->arg()->isa<Lam>();
            if (!body || !body->is_set()) return {};
            u64 total = 0;
            for (auto point : plan.points) {
                if (++work_ > 65536) return {};
                importer_.push();
                if (auto var = body->has_var()) {
                    importer_.bind_old(var);
                    importer_.map(var, importer_.rewrite(point));
                }
                auto part = region(occurrence, reader, whole, i + 1);
                importer_.pop();
                if (!part || total > std::numeric_limits<u64>::max() - *part) return {};
                total += *part;
            }
            return total;
        }
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
        DefVec origins(*rank, world_.bot(world_.type_nat()));
        auto axis = [&](this auto&& axis, size_t d) -> std::optional<u64> {
            if (d == *rank) {
                if (++work_ > 65536) return {};
                importer_.push();
                if (auto var = fn->has_var()) {
                    importer_.bind_old(var);
                    importer_.map(var, world_.tuple({world_.tuple(origins), world_.tuple(extents)}));
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
            if (enumerate_) {
                u64 total = 0;
                for (u64 off = 0; off < *s;) {
                    auto extent = std::min(k, *s - off);
                    origins[d]  = world_.lit_nat(off);
                    extents[d]  = world_.lit_nat(extent);
                    auto part   = axis(d + 1);
                    if (!part || total > std::numeric_limits<u64>::max() - *part) return {};
                    total += *part;
                    off += extent;
                }
                return total;
            }
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
    bool enumerate_;
    size_t work_ = 0;
};

std::optional<u64> outer_axis(const Def* producer, Rewriter& importer) {
    auto app = graph::map_reduce(producer);
    if (!app) return {};
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || !ranks->ro || ranks->ro > 8) return {};
    auto& w    = importer.world();
    auto value = importer.rewrite(producer);
    auto root  = w.app(w.app(w.annex<group_root>(), value->type()), value);
    if (root != w.lit_ff()) return {};
    auto record = w.app(w.annex<tile_sched>(), importer.rewrite(mr.sched));
    auto vdim   = Lit::isa(record->proj(2, 0));
    if (!vdim || !*vdim) return {};
    auto shape = importer.rewrite(mr.So);
    for (u64 d = 0; d != ranks->ro; ++d) {
        auto extent = Lit::isa(shape->proj(ranks->ro, d));
        if (!extent || !*extent) return {};
    }
    return 0;
}

class GroupPolicy {
public:
    GroupPolicy(const pattern::GraphView& source, Rewriter& importer)
        : source_(source)
        , importer_(importer) {}

    bool grouped(const Def* producer) {
        if (auto i = grouped_.find(producer); i != grouped_.end()) return i->second;
        grouped_[producer] = false;
        if (!outer_axis(producer, importer_)) return false;
        for (auto use : source_.data_uses(producer))
            if (Axm::isa<tensor::slice>(use.consumer) || Axm::isa<slice_box>(use.consumer))
                for (const auto& occurrence : source_.occurrences(use.consumer))
                    if (!occurrence.regions.empty()) return false;
        for (auto op : source_.operations())
            for (auto region : source_.isa_operation(op)->regions)
                if (auto var = region.body->has_var(); var && producer->has_free_var(var)) return false;
        auto inputs               = producers(producer);
        auto eligible             = !inputs.empty() && std::ranges::none_of(inputs, [&](auto p) { return reaches(p); });
        return grouped_[producer] = eligible;
    }

    bool terminal(const Def* producer) {
        for (auto use : source_.data_uses(producer))
            if (graph::map_reduce(use.consumer) && grouped(use.consumer)) return false;
        return true;
    }

private:
    DefVec producers(const Def* producer) const {
        DefVec result;
        if (auto op = source_.isa_operation(producer))
            for (auto input : op->inputs)
                if (graph::map_reduce(input.value)) result.push_back(input.value);
        return result;
    }

    bool reaches(const Def* producer) {
        if (auto i = reaches_.find(producer); i != reaches_.end()) return i->second;
        reaches_[producer] = true;
        auto& w            = importer_.world();
        auto value         = importer_.rewrite(producer);
        if (w.app(w.app(w.annex<group_root>(), value->type()), value) != w.lit_ff()) return true;
        if (grouped(producer)) return reaches_[producer] = false;
        auto inputs               = producers(producer);
        return reaches_[producer] = std::ranges::any_of(inputs, [&](auto p) { return reaches(p); });
    }

    const pattern::GraphView& source_;
    Rewriter& importer_;
    DefMap<bool> grouped_, reaches_;
};

} // namespace

const Def* normalize_group_build(const Def*, const Def*, const Def* arg) {
    auto [producer, size] = arg->projs<2>();
    auto cap              = Lit::isa(size);
    if (!cap || !*cap) return nullptr;
    auto& w = arg->world();
    Rewriter identity(w);
    identity.map(producer, producer);
    auto axis = outer_axis(producer, identity);
    if (!axis) return nullptr;
    auto mr   = btensor::MapRedPost::of(producer->as<App>()->callee()->as<App>());
    auto rank = *Lit::isa(mr.Ro);
    DefVec tile(rank, [&](size_t d) {
        auto n = *Lit::isa(mr.So->proj(rank, d));
        return w.lit_nat(d == *axis ? std::min(n, *cap) : n);
    });
    auto builder = w.app(w.annex<group_impl>(), {mr.Tp, mr.Ro, mr.So});
    return w.app(builder, {producer, w.tuple(tile)});
}

pattern::WitnessResult grouping_candidate(const pattern::GraphView& source, Defs inputs, World& target) {
    using pattern::CheckResult;
    auto app = graph::map_reduce(inputs[0]);
    if (!app) return {CheckResult::False, nullptr, {}};
    auto cap = Lit::isa(inputs[1]);
    auto mr  = btensor::MapRedPost::of(app->callee()->as<App>());
    if (!cap || !*cap || !source.is_program() || !pure_scalar(mr.comb) || !pure_scalar(mr.post))
        return {CheckResult::Unknown, nullptr, "grouping needs a literal tile size and pure scalar callbacks"};
    auto scratch = target.inherit();
    Rewriter importer(*scratch);
    GroupPolicy policy(source, importer);
    auto selected = policy.grouped(app) && policy.terminal(app);
    return {CheckResult::True, target.lit_bool(selected), {}};
}

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
            if (!value) {
                TileCost exact(*scratch, true);
                value = exact.count(occurrence, use.consumer, *whole);
            }
            if (!value || total > std::numeric_limits<u64>::max() - *value)
                return {CheckResult::Unknown, nullptr, "fusion occurrence has unknown trips, extents or overflow"};
            total += *value;
        }
    }
    if (!found) return {CheckResult::Unknown, nullptr, "fusion reader is outside the frozen data-use index"};
    return {CheckResult::True, target.lit_bool(total <= *limit), {}};
}

} // namespace mim::plug::tensor
