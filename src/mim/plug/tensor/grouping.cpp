#include <limits>

#include <mim/rewrite.h>

#include <mim/pattern/witness.h>
#include <mim/phase/graph_rule_rewrite.h>
#include <mim/phase/rule_rewrite.h>

#include "mim/plug/tensor/phase/group_rules.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

std::optional<u64> region_scope_budget(const pattern::GraphView&, const Def*);
bool group_fusion_producer(const Def*);

namespace {

std::optional<u64> mul(std::optional<u64> a, std::optional<u64> b) {
    if (!a || !b || (*b && *a > std::numeric_limits<u64>::max() / *b)) return {};
    return *a * *b;
}

bool scalar_operator(const Axm* axm) {
    auto name = axm->sym().view();
    return name.starts_with("core.wrap.") || name.starts_with("core.nat.") || name.starts_with("core.icmp.")
        || name.starts_with("core.ncmp.") || name.starts_with("core.conv.") || name.starts_with("core.bit")
        || name.starts_with("core.extrema.") || name == "core.select" || name.starts_with("math.arith.")
        || name.starts_with("math.cmp.") || name.starts_with("math.extrema.") || name.starts_with("math.exp.")
        || name.starts_with("math.rt.") || name.starts_with("math.conv.");
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
            if (!scalar_operator(axm)) return false;
        }
        if (def->is_mutable()) return false;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return true;
}

namespace {

struct InputRegion {
    const App* app;
    DefVec args;
    size_t input;

    const Def* values() const { return args[input]; }
    const Def* body() const { return args.back(); }
};

std::optional<InputRegion> input_region(const Def* def) {
    if (auto app = Axm::isa<tiled>(def)) {
        auto args = app->uncurry_args<5>();
        return InputRegion{
            app,
            {args.begin(), args.end()},
            3
        };
    }
    if (auto app = def->isa<App>(); app && (Axm::isa<tiled_n>(app) || Axm::isa<tiled_multi>(app))) {
        auto args = app->uncurry_args<6>();
        return InputRegion{
            app,
            {args.begin(), args.end()},
            4
        };
    }
    if (auto app = Axm::isa<tile_fold>(def)) {
        auto args = app->uncurry_args<6>();
        return InputRegion{
            app,
            {args.begin(), args.end()},
            4
        };
    }
    return {};
}

void input_leaves(const Def* value, DefVec& leaves) {
    if (auto tuple = value->isa<Tuple>())
        for (auto op : tuple->ops())
            input_leaves(op, leaves);
    else
        leaves.push_back(value);
}

const Def* input_tree(const Def* shape, Defs leaves, size_t& cursor) {
    if (auto tuple = shape->isa<Tuple>()) {
        DefVec fields;
        for (auto op : tuple->ops())
            fields.push_back(input_tree(op, leaves, cursor));
        return shape->world().tuple(fields);
    }
    return leaves[cursor++];
}

void input_parameters(const Def* shape, const Def* parameter, DefVec& fields) {
    if (auto tuple = shape->isa<Tuple>())
        for (size_t i = 0; i != tuple->num_ops(); ++i)
            input_parameters(tuple->op(i), parameter->proj(tuple->num_ops(), i), fields);
    else
        fields.push_back(parameter);
}

const Rule* region_rule(const InputRegion& region, size_t slot, u64 mode, u64 budget) {
    auto& w = region.app->world();
    DefVec leaves;
    input_leaves(region.values(), leaves);
    auto producer = graph::map_reduce(leaves[slot]);
    if (!producer) return nullptr;
    auto vars = DefVec();
    for (auto var : region.app->free_vars())
        vars.push_back(var);
    auto count  = vars.size() + leaves.size() + 1;
    auto domain = w.mut_sigma(count);
    VarRewriter types(w);
    for (size_t i = 0; i != vars.size(); ++i) {
        domain->set(i, types.rewrite(vars[i]->type()));
        types.add(vars[i]->as<Var>(), domain->var()->proj(count, i));
    }
    for (size_t i = 0; i != leaves.size(); ++i)
        domain->set(vars.size() + i, types.rewrite((i == slot ? producer->arg() : leaves[i])->type()));
    domain->set(count - 1, types.rewrite(region.body()->type()));
    auto rule   = w.mut_rule(w.reform(domain))->set("fuseRegionInput");
    auto fields = rule->vars(count);
    VarRewriter values(w);
    for (size_t i = 0; i != vars.size(); ++i)
        values.add(vars[i]->as<Var>(), fields[i]);
    auto args   = values.Rewriter::rewrite(region.args);
    auto callee = values.rewrite(producer->callee());
    DefVec inputs;
    for (size_t i = 0; i != leaves.size(); ++i)
        inputs.push_back(fields[vars.size() + i]);
    auto moved         = inputs[slot];
    inputs[slot]       = w.app(callee, moved);
    size_t cursor      = 0;
    args[region.input] = input_tree(region.values(), inputs, cursor);
    args.back()        = fields.back();
    const Def* lhs     = App::uncurry_callee(region.app);
    for (auto arg : args)
        lhs = w.app(lhs, arg);
    inputs[slot]   = moved;
    cursor         = 0;
    auto ins       = input_tree(region.values(), inputs, cursor);
    auto body_type = fields.back()->type()->as<Pi>();
    auto pi        = w.mut_pi(w.type());
    pi->set_dom(body_type->dom());
    pi->set_codom(w.pi(ins->type(), body_type->reduce(pi->var())->as<Pi>()->codom()));
    auto body  = w.mut_lam(pi);
    auto inner = w.mut_lam(pi->reduce(body->var())->as<Pi>());
    DefVec parameters;
    input_parameters(ins, inner->var(), parameters);
    parameters[slot] = w.app(callee, parameters[slot]);
    cursor           = 0;
    auto original    = input_tree(region.values(), parameters, cursor);
    inner->set(true, w.app(w.app(fields.back(), body->var()), original));
    body->set(true, inner);
    args[region.input - 1] = ins->type();
    args[region.input]     = ins;
    args.back()            = body;
    const Def* rhs         = App::uncurry_callee(region.app);
    for (auto arg : args)
        rhs = w.app(rhs, arg);
    inputs[slot] = w.app(callee, moved);
    auto guard   = w.app(w.annex<region_fusion_ok>(), {inputs[slot]->type(), lhs->type(), fields.back()->type()});
    guard = w.app(guard, {inputs[slot], lhs, fields.back(), w.lit_nat(slot), w.lit_nat(mode), w.lit_nat(budget)});
    rule->set(lhs, rhs, guard);
    return rule;
}

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

class RegionCoordinates {
public:
    explicit RegionCoordinates(World& w)
        : w_(w) {}

    DefVec points;

    bool collect(const InputRegion& call) {
        if (Axm::isa<tiled>(call.app) || Axm::isa<tiled_multi>(call.app)) {
            auto plan  = grid_tile_plan(call.app);
            auto shape = plan.shape, tiles = plan.tiles;
            auto rank = Lit::isa(shape->arity());
            if (!rank || *rank > 8) return false;
            DefVec off(*rank), ext(*rank);
            return grid(shape, tiles, off, ext, 0);
        }
        if (Axm::isa<tiled_n>(call.app)) {
            auto plan = multi_tile_plan(call.app);
            for (auto point : plan.points)
                points.push_back(point);
            return !points.empty() && points.size() <= 65536;
        }
        auto [n, ks] = call.args[1]->projs<2>();
        auto count = Lit::isa(n), stride = Lit::isa(ks);
        if (!count || !stride || *count > 65536) return false;
        auto acc = w_.mut_lam(call.args[0], w_.type_nat())->var();
        for (u64 i = 0; i != *count; ++i) {
            if (*stride && i > std::numeric_limits<u64>::max() / *stride) return false;
            points.push_back(w_.tuple({acc, w_.lit_nat(i * *stride)}));
        }
        return true;
    }

private:
    bool grid(const Def* shape, const Def* tiles, DefVec& off, DefVec& ext, size_t d) {
        if (points.size() >= 65536) return false;
        if (d == off.size()) {
            points.push_back(w_.tuple({w_.tuple(off), w_.tuple(ext)}));
            return true;
        }
        auto n = Lit::isa(shape->proj(off.size(), d)), k = Lit::isa(tiles->proj(off.size(), d));
        if (!n || !k || !*k) return false;
        for (u64 i = 0; i < *n;) {
            auto size = std::min(*k, *n - i);
            off[d]    = w_.lit_nat(i);
            ext[d]    = w_.lit_nat(size);
            if (!grid(shape, tiles, off, ext, d + 1)) return false;
            i += size;
        }
        return true;
    }

    World& w_;
};

class RegionFusionProof {
public:
    RegionFusionProof(World& w, const Def* producer, const Def* parameter, u64 mode, u64 limit)
        : w_(w)
        , producer_(producer)
        , parameter_(parameter)
        , mode_(mode)
        , limit_(limit) {}

    bool region(const InputRegion& call, const Def* inputs) {
        if (++work_ > 65536) return false;
        if (Axm::isa<tile_fold>(call.app) && !visit(call.args[2])) return false;
        RegionCoordinates grid(w_);
        if (!grid.collect(call)) return false;
        for (auto point : grid.points)
            if (!body(call.body(), point, inputs)) return false;
        if (!verify_ && !Axm::isa<tiled_n>(call.app)) {
            DefSet extents;
            verify_ = true;
            for (auto point : grid.points) {
                auto [first, second] = point->projs<2>();
                auto grid            = Axm::isa<tiled>(call.app) || Axm::isa<tiled_multi>(call.app);
                if (grid && !extents.insert(second).second) continue;
                auto variable   = w_.mut_lam(grid ? first->type() : second->type(), w_.type_nat())->var();
                auto coordinate = grid ? w_.tuple({variable, second}) : w_.tuple({first, variable});
                if (!body(call.body(), coordinate, inputs)) {
                    verify_ = false;
                    return false;
                }
            }
            verify_ = false;
        }
        return true;
    }

    u64 reads() const { return reads_; }

private:
    bool body(const Def* fn, const Def* coordinate, const Def* inputs) {
        if (++work_ > 65536) return false;
        return visit(w_.app(w_.app(fn, coordinate), inputs));
    }

    bool read(const App* app) {
        const Def *off, *ext, *cap;
        auto [meta, shape, params, source] = app->uncurry_args<4>();
        if (source != parameter_) return false;
        if (Axm::isa<slice>(app)) {
            if (mode_ == 1) return false;
            auto [o, step, e] = params->projs<3>();
            auto rank         = Lit::isa(e->arity());
            if (!rank) return false;
            for (u64 d = 0; d != *rank; ++d)
                if (Lit::isa(step->proj(*rank, d)) != 1) return false;
            off = o;
            ext = cap = e;
        } else {
            if (!mode_) return false;
            std::tie(off, ext, cap) = params->projs<3>();
        }
        auto box    = Axm::isa<slice_box>(app) != nullptr;
        auto result = producer_tile(producer_, off, ext, box ? cap : nullptr);
        if (!result.value) {
            w_.log().d("tensor region fusion cannot tile {}: {}", producer_, result.reason);
            return false;
        }
        if (!graph::map_reduce(result.value) && !Axm::isa<splat>(result.value)) return false;
        if (verify_) return true;
        auto rank                 = Lit::isa(ext->arity());
        std::optional<u64> volume = 1;
        if (!rank) return false;
        for (u64 d = 0; d != *rank; ++d)
            volume = mul(volume, Lit::isa(ext->proj(*rank, d)));
        if (!volume || *volume > limit_ - total_) return false;
        total_ += *volume;
        ++reads_;
        return true;
    }

    bool visit(const Def* root) {
        DefVec pending{root};
        DefSet seen;
        for (size_t i = 0; i != pending.size(); ++i) {
            if (++work_ > 65536) return false;
            auto def = pending[i];
            if (!seen.insert(def).second) continue;
            if (def == parameter_) return false;
            if (!def->has_free_var(parameter_->as<Var>())) continue;
            if (auto call = input_region(def)) {
                if (!region(*call, call->values())) return false;
                continue;
            }
            if (auto app = def->isa<App>(); app && (Axm::isa<slice>(app) || Axm::isa<slice_box>(app))) {
                if (app->arg() == parameter_) {
                    if (!read(app)) return false;
                    continue;
                }
            }
            if (def->isa<Lam>() || def->isa<Pi>() || def->isa<Axm>() || def->isa<Var>()) return false;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        return true;
    }

    World& w_;
    const Def *producer_, *parameter_;
    u64 mode_, limit_, total_ = 0, reads_ = 0;
    size_t work_ = 0;
    bool verify_ = false;
};

class RegionFusionSamples {
public:
    RegionFusionSamples(World& w,
                        const InputRegion& region,
                        size_t slot,
                        RegionFusionProof& proof,
                        const Def* parameter)
        : w_(w)
        , region_(region)
        , slot_(slot)
        , proof_(proof)
        , parameter_(parameter) {}

    bool check(std::span<const Def* const> ancestors, size_t index = 0) {
        if (++work_ > 65536) return false;
        pattern::ParameterImporter importer(w_);
        for (auto [old, value] : bindings_)
            importer.map(old, value);
        if (index == ancestors.size()) {
            auto call = input_region(importer.rewrite(region_.app));
            if (!call) return false;
            DefVec leaves;
            input_leaves(call->values(), leaves);
            if (slot_ >= leaves.size()) return false;
            leaves[slot_] = parameter_;
            size_t cursor = 0;
            return proof_.region(*call, input_tree(call->values(), leaves, cursor));
        }
        auto source = input_region(ancestors[index]);
        if (!source) return false;
        auto imported = input_region(importer.rewrite(source->app));
        if (!imported) return false;
        RegionCoordinates grid(w_);
        if (!grid.collect(*imported)) return false;
        auto body  = source->body()->isa<Lam>();
        auto inner = body && body->is_set() ? body->body()->isa<Lam>() : nullptr;
        if (!inner || !inner->is_set()) return false;
        auto size = bindings_.size();
        for (auto point : grid.points) {
            if (body->has_var()) bindings_.emplace_back(body->has_var(), point);
            if (inner->has_var()) bindings_.emplace_back(inner->has_var(), imported->values());
            if (!check(ancestors, index + 1)) return false;
            bindings_.resize(size);
        }
        return true;
    }

private:
    World& w_;
    InputRegion region_;
    size_t slot_;
    RegionFusionProof& proof_;
    const Def* parameter_;
    std::vector<std::pair<const Def*, const Def*>> bindings_;
    size_t work_ = 0;
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
                if (auto inner = body->body()->isa<Lam>())
                    if (auto var = inner->has_var()) {
                        importer_.bind_old(var);
                        importer_.map(var, importer_.rewrite(tiled->uncurry_args<6>()[4]));
                    }
                auto part = region(occurrence, reader, whole, i + 1);
                importer_.pop();
                if (!part || total > std::numeric_limits<u64>::max() - *part) return {};
                total += *part;
            }
            return total;
        }
        if (auto fold = Axm::isa<tile_fold>(call)) {
            auto [type, blocks, init, it, ins, body] = fold->uncurry_args<6>();
            auto n                                   = Lit::isa(importer_.rewrite(blocks->proj(2, 0)));
            return mul(n, region(occurrence, reader, whole, i + 1));
        }
        auto tiled = call->isa<App>();
        if (!tiled || (!Axm::isa<tensor::tiled>(call) && !Axm::isa<tiled_multi>(call))) return {};
        auto multiple = Axm::isa<tiled_multi>(call) != nullptr;
        auto args     = input_region(tiled)->args;
        auto rank     = Lit::isa(args[0]->proj(multiple ? 4 : 2, 1));
        auto ins      = args[args.size() - 2];
        auto fn       = args.back()->isa<Lam>();
        if (!rank || *rank > 8 || !fn || !fn->is_set()) return {};
        auto shape = args[1]->proj(multiple ? 3 : 2, multiple ? 1 : 0);
        auto tile  = args[1]->proj(multiple ? 3 : 2, multiple ? 2 : 1);
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
                if (auto inner = fn->body()->isa<Lam>())
                    if (auto var = inner->has_var()) {
                        importer_.bind_old(var);
                        importer_.map(var, importer_.rewrite(ins));
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
    target.log().d("tensor fusion budget {}: work={} limit={}", producer, total, *limit);
    return {CheckResult::True, target.lit_bool(total <= *limit), {}};
}

pattern::WitnessResult region_fusion_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    using pattern::CheckResult;
    auto producer = graph::map_reduce(inputs[0]);
    auto call     = input_region(inputs[1]);
    auto slot = Lit::isa(inputs[3]), mode = Lit::isa(inputs[4]), budget = Lit::isa(inputs[5]);
    if (!producer || !call || !slot || !mode || !budget || !*budget)
        return {CheckResult::Unknown, nullptr, "region fusion needs a frozen producer and literal policy"};
    if (inputs[2] != call->body())
        return {CheckResult::False, nullptr, "region fusion callback does not match the frozen region"};
    if (!source.is_contained(producer))
        return {CheckResult::Unknown, nullptr, "region fusion producer is outside the frozen graph"};
    if (source.complete_data_uses(producer) != CheckResult::True)
        return {CheckResult::Unknown, nullptr, "region fusion producer has incomplete data uses"};
    if (*mode == 2) {
        budget = region_scope_budget(source, call->app);
        if (!budget) return {CheckResult::False, nullptr, "region fusion is outside a requested group"};
    }
    DefMap<size_t> readers;
    for (auto use : source.data_uses(producer)) {
        auto reader = input_region(use.consumer);
        if (!reader || use.group != reader->input)
            return {CheckResult::False, nullptr, "region fusion retains a producer with another reader"};
        DefVec leaves;
        input_leaves(reader->values(), leaves);
        auto found = std::ranges::find(leaves, producer);
        if (found == leaves.end() || std::ranges::count(leaves, producer) != 1)
            return {CheckResult::False, nullptr, "region fusion requires one unaliased producer input"};
        readers.emplace(reader->app, size_t(found - leaves.begin()));
        if (*mode == 2) {
            auto cap = region_scope_budget(source, reader->app);
            if (!cap) return {CheckResult::False, nullptr, "region fusion has an unscoped reader"};
            budget = std::min(*budget, *cap);
        }
    }
    if (auto reader = readers.find(call->app); reader == readers.end() || reader->second != *slot)
        return {CheckResult::False, nullptr, "region fusion input does not match the frozen producer"};
    auto mr    = btensor::MapRedPost::of(producer->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || !pure_scalar(mr.comb) || !pure_scalar(mr.post))
        return {CheckResult::False, nullptr, "region fusion requires pure producer callbacks"};
    auto scratch = target.inherit();
    pattern::ParameterImporter importer(*scratch);
    auto p = importer.rewrite(producer);
    if (*mode == 2 && !group_fusion_producer(p))
        return {CheckResult::False, nullptr, "region fusion cannot cross a BLIS, view or placed root"};
    if (*mode == 0 && scratch->app(scratch->app(scratch->annex<group_root>(), p->type()), p) != scratch->lit_ff())
        return {CheckResult::False, nullptr, "region fusion cannot move a producer root"};
    auto shape                = importer.rewrite(mr.So);
    std::optional<u64> volume = 1;
    for (u64 d = 0; d != ranks->ro; ++d)
        volume = mul(volume, Lit::isa(shape->proj(ranks->ro, d)));
    auto limit = mul(volume, budget);
    if (!limit) return {CheckResult::Unknown, nullptr, "region fusion has an unknown or overflowing budget"};
    auto sentinel = scratch->mut_lam(p->type(), scratch->type_nat());
    RegionFusionProof proof(*scratch, p, sentinel->var(), *mode, *limit);
    bool accepted = true;
    for (auto [reader, position] : readers) {
        RegionFusionSamples samples(*scratch, *input_region(reader), position, proof, sentinel->var());
        accepted = accepted && !source.occurrences(reader).empty();
        for (auto occurrence : source.occurrences(reader))
            accepted = accepted && samples.check(occurrence.regions);
    }
    accepted = accepted && proof.reads();
    return {CheckResult::True, target.lit_bool(accepted), {}};
}

namespace phase {

void FuseRegionInputs::apply(const App* app) {
    auto [mode, budget, graph] = app->args<3>();
    if (!Lit::isa(mode) || Lit::as(mode) > 2 || !Lit::isa(budget) || !Lit::isa(graph))
        app->blame("tensor.fuse_region_inputs requires a literal mode (0, 1 or 2), budget and backend").bail();
    mode_   = Lit::as(mode);
    budget_ = Lit::as(budget);
    graph_  = Lit::as(graph);
}

std::unique_ptr<Phase> FuseRegionInputs::recreate() {
    auto phase     = std::make_unique<FuseRegionInputs>(world(), std::string(name()));
    phase->mode_   = mode_;
    phase->budget_ = budget_;
    phase->graph_  = graph_;
    return phase;
}

void FuseRegionInputs::start() {
    std::vector<const Rule*> rules, reads;
    std::vector<const Axm*> pure;
    {
        pattern::GraphView source(world(), driver().graph_operations());
        for (auto node : source.nodes())
            if (auto axm = node->isa<Axm>()) {
                auto name = axm->sym().view();
                if (scalar_operator(axm) || name.starts_with("affine.") || name == "tensor.splat"
                    || name == "tensor.tile_of" || name == "tensor.tile_of_box")
                    pure.push_back(axm);
            }
        DefVec regions;
        for (auto op : source.operations())
            if (input_region(op)) regions.push_back(op);
        DefSet signatures, types;
        for (auto def : regions) {
            auto region = *input_region(def);
            DefVec leaves;
            input_leaves(region.values(), leaves);
            for (size_t i = 0; i != leaves.size(); ++i)
                if (auto producer = graph::map_reduce(leaves[i])) {
                    auto mr   = btensor::MapRedPost::of(producer->callee()->as<App>());
                    auto meta = world().tuple({mr.Tp, mr.Ro});
                    if (!meta->is_closed()) continue;
                    if (types.insert(meta).second) {
                        if (mode_ == 2) {
                            reads.push_back(world().app(world().annex<group_slice_rule>(), meta)->as<Rule>());
                            reads.push_back(world().app(world().annex<group_box_rule>(), meta)->as<Rule>());
                        } else {
                            auto factory
                                = mode_ ? world().annex<fusion_box_rule>() : world().annex<fusion_slice_rule>();
                            reads.push_back(world().app(factory, {mr.Tp, mr.Ro, world().lit_nat(budget_)})->as<Rule>());
                        }
                    }
                    auto key          = region.args;
                    key[region.input] = world().tuple({producer->callee(), world().lit_nat(i)});
                    key.back()        = region.body()->type();
                    key.push_back(App::uncurry_callee(region.app));
                    if (signatures.insert(world().tuple(key)).second)
                        if (auto rule = region_rule(region, i, mode_, budget_)) rules.push_back(rule);
                }
        }
    }
    if (rules.empty()) return;
    if (graph_) {
        for (auto name : {"tensor.tiled", "tensor.tiled_multi", "tensor.tile_fold", "tensor.tiled_n",
                          "tensor.map_reduce_post", "tensor.tile_of", "tensor.tile_of_box", "btensor.clip_sched"})
            pure.push_back(world().annex(driver().sym(name))->as<Axm>());
        GraphRuleRewrite phase(world(), std::move(rules), pure, true);
        GraphRuleRewrite inner(world(), std::move(reads), std::move(pure), true);
        phase.run();
        auto changed = phase.todo();
        if (changed) inner.run();
        invalidate(changed || inner.todo());
    } else {
        RuleRewrite phase(world(), std::move(rules));
        RuleRewrite inner(world(), std::move(reads));
        phase.run();
        auto changed = phase.todo();
        if (changed) inner.run();
        invalidate(changed || inner.todo());
    }
}

} // namespace phase

} // namespace mim::plug::tensor
