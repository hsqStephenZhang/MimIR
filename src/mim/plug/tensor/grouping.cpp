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

struct YieldRead {
    const App* app;
    const Def *off, *ext, *cap;
};

class YieldReads {
public:
    YieldReads(const Def* producer, const Def* root, const Def* owned_extent) {
        DefVec pending{root};
        DefSet seen;
        for (size_t i = 0; i != pending.size(); ++i) {
            if (i > 65536) return;
            auto def = pending[i];
            if (!seen.insert(def).second) continue;
            if (def == producer) {
                auto type = TensorType::isa(producer->type());
                if (!type || type->s != owned_extent) return;
                reads.push_back({producer->as<App>(), producer->world().pack(type->r, producer->world().lit_nat_0()),
                                 type->s, type->s});
                continue;
            }
            if (auto app = def->isa<App>();
                app && (Axm::isa<slice>(app) || Axm::isa<slice_box>(app)) && app->arg() == producer) {
                auto [meta, shape, params, source] = app->uncurry_args<4>();
                auto [off, middle, last]           = params->projs<3>();
                if (Axm::isa<slice>(app)) {
                    auto rank = Lit::isa(last->arity());
                    if (!rank) return;
                    for (u64 d = 0; d != *rank; ++d)
                        if (Lit::isa(middle->proj(*rank, d)) != 1) return;
                    reads.push_back({app, off, last, last});
                } else {
                    reads.push_back({app, off, middle, last});
                }
                continue;
            }
            if (def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
            if (auto lam = def->isa<Lam>(); lam && (!lam->is_set() || lam->is_external())) continue;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        complete = true;
    }

    fe::Vector<YieldRead> reads;
    bool complete = false;
};

ProducerTile
yield_tiles_value(const Def* producer, const Def* off, const Def* ext, const Def* value, bool proof, u64& work) {
    YieldReads uses(producer, value, ext);
    if (!uses.complete || uses.reads.size() != 1) return {nullptr, "yield needs one shared tileable slice/box read"};
    auto& w   = producer->world();
    auto rank = Lit::isa(ext->arity());
    if (!rank) return {nullptr, "yield needs a literal producer rank"};
    Rewriter rw(w);
    for (auto var : value->free_vars())
        rw.map(var, var);
    const Def* owned = nullptr;
    for (auto read : uses.reads) {
        auto tile = producer_tile(producer, read.off, read.ext,
                                  Axm::isa<slice_box>(read.app) || input_region(value) ? read.cap : nullptr);
        if (!tile.value) return tile;
        rw.map(read.app, tile.value);
        if (proof) {
            std::optional<u64> cells = 1;
            for (u64 d = 0; d != *rank; ++d)
                cells = mul(cells, Lit::isa(read.ext->proj(*rank, d)));
            if (!cells || work > std::numeric_limits<u64>::max() - *cells)
                return {nullptr, "yield has unknown or overflowing tile work"};
            work += *cells;
        }
        if (!owned) {
            // The staged witness proves containment before this private recipe is instantiated.
            DefVec local;
            for (u64 d = 0; d != *rank; ++d) {
                auto start = off->proj(*rank, d), size = ext->proj(*rank, d);
                auto base = read.off->proj(*rank, d), length = read.ext->proj(*rank, d);
                if (proof) {
                    auto a = Lit::isa(start), b = Lit::isa(base), n = Lit::isa(size), m = Lit::isa(length);
                    if (!a || !b || !n || !m || *a < *b || *a - *b > *m || *n > *m - (*a - *b))
                        return {nullptr, "yield owned box is not contained in its first producer read"};
                }
                local.push_back(w.call(core::nat::sub, Defs{start, base}));
            }
            if (off == read.off && ext == read.ext)
                owned = tile.value;
            else {
                auto mr    = btensor::MapRedPost::of(producer->as<App>()->callee()->as<App>());
                auto slice = w.app(w.app(w.annex<tensor::slice_box>(), {mr.Tp, mr.Ro}), read.ext);
                owned      = w.app(w.app(slice, {w.tuple(local), ext, ext}), tile.value);
            }
        }
    }
    return {w.tuple({owned, rw.rewrite(value)}), {}};
}

class YieldAxes {
public:
    YieldAxes(const GridTilePlan& plan, const btensor::MapRedPost& producer)
        : plan_(plan)
        , producer_(producer)
        , rank_(Lit::as(producer.Ro))
        , used_(rank_, false) {
        collect(0);
    }
    DefVec choices;

private:
    void collect(u64 d) {
        if (choices.size() >= 64) return;
        auto& w = producer_.So->world();
        if (d == plan_.rank) {
            choices.push_back(w.tuple(axes_));
            return;
        }
        for (u64 a = 0; a != rank_; ++a)
            if (!used_[a] && Check::test_alpha(plan_.shape->proj(plan_.rank, d), producer_.So->proj(rank_, a))) {
                used_[a] = true;
                axes_.push_back(w.lit_idx(rank_, a));
                collect(d + 1);
                axes_.pop_back();
                used_[a] = false;
            }
    }
    const GridTilePlan& plan_;
    const btensor::MapRedPost& producer_;
    u64 rank_;
    fe::Vector<bool> used_;
    DefVec axes_;
};

class RowInputs : public Rewriter {
public:
    RowInputs(World& w, const Def* source, const Def* rows, const Def* origin)
        : Rewriter(w)
        , source_(source)
        , rows_(rows)
        , origin_(origin) {}

private:
    const Def* expand(const Def* value) {
        auto type = value->type();
        if (!TensorType::isa(type) && (type->isa<Sigma>() || type->isa<Arr>())) {
            auto n = Lit::isa(type->arity());
            if (n) return world().tuple(DefVec(*n, [&](size_t i) { return expand(value->proj(*n, i)); }));
        }
        return rewrite(value);
    }

    void leaves(const Def* shape, const Def* parameter, DefVec& values) {
        auto type = shape->type();
        if (!TensorType::isa(type) && (type->isa<Sigma>() || type->isa<Arr>())) {
            if (auto n = Lit::isa(type->arity())) {
                for (u64 i = 0; i != *n; ++i)
                    leaves(shape->proj(*n, i), parameter->proj(*n, i), values);
                return;
            }
        }
        values.push_back(parameter);
    }

    const Def* rewrite(const Def* def) override {
        if (def == source_) return rows_;
        if (auto mapped = lookup(def)) return mapped;
        if (auto var = def->isa<Var>()) {
            if (auto binder = lookup(var->binder())) return world().var(binder->as_mut());
            return def;
        }
        return Rewriter::rewrite(def);
    }

    const Def* rewrite_imm_App(const App* app) override {
        auto& w = world();
        if ((Axm::isa<slice>(app) || Axm::isa<slice_box>(app)) && app->arg() == source_) {
            auto [meta, shape, params, source] = app->uncurry_args<4>();
            auto [off, middle, last]           = params->projs<3>();
            auto [row, col]                    = off->projs<2>();
            auto offset = w.tuple({w.call(core::nat::sub, Defs{rewrite(row), origin_}), rewrite(col)});
            auto local  = TensorType::isa(rows_->type())->s;
            auto callee = w.app(w.app(App::uncurry_callee(app), meta), local);
            return w.app(w.app(callee, {offset, rewrite(middle), rewrite(last)}), rows_);
        }
        if (auto region = input_region(app)) {
            auto args        = region->args;
            auto original    = region->values();
            auto values      = expand(original);
            const Def* build = App::uncurry_callee(app);
            for (size_t i = 0; i + 1 != args.size(); ++i)
                build = w.app(build, i == region->input - 1 ? values->type()
                                     : i == region->input   ? values
                                                            : rewrite(args[i]));
            auto body  = w.mut_lam(build->type()->as<Pi>()->dom()->as<Pi>());
            auto inner = w.mut_lam(body->type()->reduce(body->var())->as<Pi>());
            DefVec old, parameters;
            leaves(original, original, old);
            leaves(values, inner->var(), parameters);
            RowInputs child(w, source_, rows_, origin_);
            for (size_t i = 0; i != old.size(); ++i)
                if (old[i] == source_ || old[i] == rows_)
                    child.rows_ = parameters[i];
                else
                    child.map(old[i], parameters[i]);
            auto value = w.app(w.app(region->body(), body->var()), original);
            inner->set(true, child.rewrite(value));
            body->set(true, inner);
            return w.app(build, body);
        }
        return Rewriter::rewrite_imm_App(app);
    }

    const Def* rewrite_imm_Tuple(const Tuple* tuple) override { return world().tuple(Rewriter::rewrite(tuple->ops())); }

    const Def *source_, *rows_, *origin_;
};

const Def* outer_rows(const App* app, size_t slot, u64 tile) {
    auto& w     = app->world();
    auto region = *input_region(app);
    auto plan   = grid_tile_plan(app);
    DefVec leaves;
    input_leaves(region.values(), leaves);
    auto source     = leaves[slot];
    auto type       = *TensorType::isa(source->type());
    auto [m, n]     = plan.shape->projs<2>();
    auto k          = type.s->proj(2, 1);
    auto build      = w.app(w.annex<tiled_multi>(), {w.lit_nat_1(), w.lit_nat_1(), plan.types, plan.ranks});
    build           = w.app(w.app(build, {plan.shapes, m, w.lit_nat(tile), plan.tail}), w.lit_idx(2, 0));
    build           = w.app(w.app(build, region.values()->type()), region.values());
    auto body       = w.mut_lam(build->type()->as<Pi>()->dom()->as<Pi>());
    auto inner      = w.mut_lam(body->type()->reduce(body->var())->as<Pi>());
    auto [off, ext] = body->vars<2>();
    DefVec parameters;
    input_parameters(region.values(), inner->var(), parameters);
    auto p     = parameters[slot];
    auto slice = w.app(w.app(w.annex<tensor::slice>(), {type.T, type.r}), type.s);
    auto rows  = w.app(w.app(slice, {w.tuple({off, w.lit_nat_0()}), w.pack(2, w.lit_nat_1()), w.tuple({ext, k})}), p);
    RowInputs crop(w, p, rows, off);
    auto args          = region.args;
    args[1]            = w.tuple({w.tuple({ext, n}), w.tuple({ext, plan.tiles->proj(2, 1)}), plan.tail});
    size_t cursor      = 0;
    args[region.input] = input_tree(region.values(), parameters, cursor);
    const Def* nested  = App::uncurry_callee(app);
    for (size_t i = 0; i + 1 != args.size(); ++i)
        nested = w.app(nested, args[i]);
    auto nbody        = w.mut_lam(nested->type()->as<Pi>()->dom()->as<Pi>());
    auto ninner       = w.mut_lam(nbody->type()->reduce(nbody->var())->as<Pi>());
    auto [noff, next] = nbody->vars<2>();
    auto [row, col]   = noff->projs<2>();
    auto global       = w.tuple({w.call(core::nat::add, Defs{off, row}), col});
    ninner->set(true, w.app(w.app(region.body(), {global, next}), ninner->var()));
    nbody->set(true, ninner);
    nested = w.app(nested, nbody);
    inner->set(true, crop.Rewriter::rewrite(nested));
    body->set(true, inner);
    return w.app(build, body);
}

const Rule* outer_rows_rule(const InputRegion& region, size_t slot, u64 tile) {
    auto& w = region.app->world();
    DefVec vars;
    for (auto var : region.app->free_vars())
        vars.push_back(var);
    auto count  = vars.size() + 2;
    auto domain = w.mut_sigma(count);
    VarRewriter types(w);
    for (size_t i = 0; i != vars.size(); ++i) {
        domain->set(i, types.rewrite(vars[i]->type()));
        types.add(vars[i]->as<Var>(), domain->var()->proj(count, i));
    }
    domain->set(count - 2, types.rewrite(region.values()->type()));
    domain->set(count - 1, types.rewrite(region.body()->type()));
    auto rule   = w.mut_rule(w.reform(domain))->set("outerRows");
    auto fields = rule->vars(count);
    VarRewriter values(w);
    for (size_t i = 0; i != vars.size(); ++i)
        values.add(vars[i]->as<Var>(), fields[i]);
    auto args          = values.Rewriter::rewrite(region.args);
    args[region.input] = fields[count - 2];
    args.back()        = fields.back();
    const Def* lhs     = App::uncurry_callee(region.app);
    for (auto arg : args)
        lhs = w.app(lhs, arg);
    auto rhs   = w.app(w.annex<split_rows>(), lhs->type());
    rhs        = w.app(rhs, {lhs, w.lit_nat(slot), w.lit_nat(tile)});
    auto guard = w.app(w.annex<outer_rows_ok>(), lhs->type());
    guard      = w.app(guard, {lhs, w.lit_nat(slot), w.lit_nat(tile)});
    return rule->set(lhs, rhs, guard);
}

const Rule* yield_rule(const InputRegion& region, size_t slot, const Def* axes, u64 budget) {
    auto& w   = region.app->world();
    auto plan = grid_tile_plan(region.app);
    DefVec leaves;
    input_leaves(region.values(), leaves);
    auto producer = graph::map_reduce(leaves[slot]);
    DefVec vars;
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
    auto rule   = w.mut_rule(w.reform(domain))->set("fuseAndYield");
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
    auto p             = w.app(callee, moved);
    inputs[slot]       = p;
    size_t cursor      = 0;
    args[region.input] = input_tree(region.values(), inputs, cursor);
    args.back()        = fields.back();
    const Def* lhs     = App::uncurry_callee(region.app);
    for (auto arg : args)
        lhs = w.app(lhs, arg);
    inputs[slot]     = moved;
    cursor           = 0;
    auto ins         = input_tree(region.values(), inputs, cursor);
    auto source_plan = grid_tile_plan(lhs->as<App>());
    auto source_mr   = btensor::MapRedPost::of(p->as<App>()->callee()->as<App>());
    DefVec result_types{source_mr.Tp}, result_ranks{source_mr.Ro}, result_shapes{source_mr.So}, mappings{axes};
    for (u64 i = 0; i != plan.count; ++i) {
        result_types.push_back(source_plan.types->proj(plan.count, i));
        result_ranks.push_back(source_plan.ranks->proj(plan.count, i));
        result_shapes.push_back(source_plan.shapes->proj(plan.count, i));
        mappings.push_back(source_plan.axes->proj(plan.count, i));
    }
    auto build = w.app(w.annex<tiled_multi>(),
                       {w.lit_nat(plan.count + 1), w.lit_nat(plan.rank), w.tuple(result_types), w.tuple(result_ranks)});
    build      = w.app(w.app(build, {w.tuple(result_shapes), source_plan.shape, source_plan.tiles, source_plan.tail}),
                       w.tuple(mappings));
    build      = w.app(w.app(build, ins->type()), ins);
    auto body  = w.mut_lam(build->type()->as<Pi>()->dom()->as<Pi>());
    auto inner = w.mut_lam(body->type()->reduce(body->var())->as<Pi>());
    DefVec parameters;
    input_parameters(ins, inner->var(), parameters);
    auto rebuilt     = w.app(callee, parameters[slot]);
    parameters[slot] = rebuilt;
    cursor           = 0;
    auto original    = input_tree(region.values(), parameters, cursor);
    auto value       = w.app(w.app(fields.back(), body->var()), original);
    auto [off, ext]  = body->vars<2>();
    GridTilePlan owned{source_mr.Tp,
                       source_mr.Ro,
                       source_mr.So,
                       source_plan.shape,
                       source_plan.tiles,
                       source_plan.tail,
                       axes,
                       ins,
                       body,
                       1,
                       plan.rank,
                       false};
    auto [start, extent] = owned.box(0, off, ext);
    auto tile = w.app(w.app(w.annex<yield_tiles>(), {source_mr.Tp, source_mr.Ro, value->type()}), source_mr.So);
    auto pair = w.app(tile, {rebuilt, start, extent, value});
    DefVec results{pair->proj(2, 0)};
    for (u64 i = 0; i != plan.count; ++i)
        results.push_back(plan.multiple ? pair->proj(2, 1)->proj(plan.count, i) : pair->proj(2, 1));
    inner->set(true, w.tuple(results));
    body->set(true, inner);
    build = w.app(build, body);
    DefVec replacements;
    for (u64 i = 0; i != plan.count; ++i)
        replacements.push_back(build->proj(plan.count + 1, i + 1));
    auto guard = w.app(w.annex<region_yield_ok>(), {p->type(), lhs->type(), fields.back()->type(), axes->type()});
    guard      = w.app(guard, {p, lhs, fields.back(), axes, w.lit_nat(slot), w.lit_nat(budget)});
    return rule->set(
        w.tuple({p, lhs}),
        w.tuple({build->proj(plan.count + 1, 0), plan.multiple ? w.tuple(replacements) : replacements.front()}), guard);
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
        auto shape = args[1]->proj(multiple ? 4 : 3, multiple ? 1 : 0);
        auto tile  = args[1]->proj(multiple ? 4 : 3, multiple ? 2 : 1);
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

const Def* normalize_yield_tiles(const Def*, const Def*, const Def* arg) {
    auto [producer, off, ext, value] = arg->projs<4>();
    if (!graph::map_reduce(producer)) return nullptr;
    u64 work = 0;
    return yield_tiles_value(producer, off, ext, value, false, work).value;
}

const Def* normalize_split_rows(const Def*, const Def*, const Def* arg) {
    auto [value, index, size] = arg->projs<3>();
    auto region               = input_region(value);
    auto slot = Lit::isa(index), tile = Lit::isa(size);
    if (!region || !Axm::isa<tiled>(value) || !slot || !tile || !*tile) return nullptr;
    DefVec leaves;
    input_leaves(region->values(), leaves);
    if (*slot >= leaves.size() || !graph::map_reduce(leaves[*slot])) return nullptr;
    return outer_rows(region->app, *slot, *tile);
}

namespace {

struct RowPolicy {
    u64 tile = 0, mr = 0, kc = 0, extra = 0;
    std::string reason;
};

RowPolicy row_policy(const InputRegion& region, size_t slot) {
    RowPolicy policy;
    auto plan = grid_tile_plan(region.app);
    DefVec leaves;
    input_leaves(region.values(), leaves);
    auto type = slot < leaves.size() ? TensorType::isa(leaves[slot]->type()) : std::nullopt;
    if (!Axm::isa<tiled>(region.app) || plan.rank != 2 || !type || Lit::isa(type->r) != 2) {
        policy.reason = "row split needs a two-dimensional consumer and producer";
        return policy;
    }
    auto shape = lit_projs(plan.shape, 2), tiles = lit_projs(plan.tiles, 2), ps = lit_projs(type->s, 2);
    if (!shape || !tiles || !ps || (*ps)[0] != (*shape)[0] || (*tiles)[0] != (*shape)[0] || slot != 0) {
        policy.reason = "row split needs an unsplit M axis and a whole-width left operand";
        return policy;
    }
    auto weights = leaves.size() > 1 ? TensorType::isa(leaves[1]->type()) : std::nullopt;
    auto ws      = weights && Lit::isa(weights->r) == 2 ? lit_projs(weights->s, 2) : std::nullopt;
    if (!ws || weights->T != type->T || (*ws)[0] != (*ps)[1] || (*ws)[1] != (*shape)[1]) {
        policy.reason = "row split needs a same-element-type K-by-N weight operand";
        return policy;
    }
    DefVec pending{region.body()};
    DefSet seen;
    u64 nr = 0;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second || def->isa<Axm>() || def->isa<Var>() || def->isa<Pi>()) continue;
        if (auto nested = Axm::isa<tiled>(def)) {
            auto micro = grid_tile_plan(nested);
            if (micro.rank == 2)
                if (auto blocks = lit_projs(micro.tiles, 2)) {
                    if (policy.mr && (policy.mr != (*blocks)[0] || nr != (*blocks)[1])) {
                        policy.reason = "row split needs one consistent BLIS micro-tile";
                        return policy;
                    }
                    policy.mr = (*blocks)[0];
                    nr        = (*blocks)[1];
                }
        }
        if (auto fold = input_region(def); fold && Axm::isa<tile_fold>(def))
            policy.kc = std::max(policy.kc, Lit::isa(fold->args[1]->proj(2, 1)).value_or(0));
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    auto& w    = region.app->world();
    auto size  = Lit::isa(w.app(w.annex(core::trait::size), type->T));
    auto denom = mul(mul(2, policy.kc ? policy.kc : (*ps)[1]), size);
    auto cache = w.driver().arg_nat("tensor", "group-cache").value_or(262144);
    auto mc    = denom && *denom && policy.mr ? cache / *denom / policy.mr * policy.mr : 0;
    if (!mc || !nr) {
        policy.reason = "row split efficiency: cache cannot hold an MR-aligned M block";
        return policy;
    }
    auto m = (*shape)[0], n = (*shape)[1], k = (*ps)[1];
    policy.tile = std::min(m, mc);
    if ((m - 1) / policy.tile >= 65536) {
        policy.reason = "row split grid exceeds the proof limit";
        return policy;
    }
    if (n > std::numeric_limits<u64>::max() - (nr - 1)) {
        policy.reason = "row split weight re-streaming arithmetic overflow";
        return policy;
    }
    auto extra = mul(mul(mul((m - 1) / policy.tile, k), (n + nr - 1) / nr * nr), size);
    auto owned = mul(mul(m, k), size);
    if (!extra || !owned || *extra > *owned) {
        policy.reason = "row split weight re-streaming exceeds one producer buffer";
        return policy;
    }
    policy.extra = *extra;
    return policy;
}

class RowReadProof {
public:
    RowReadProof(World& w, const Def* source, u64 first, u64 rows, u64 width)
        : w_(w)
        , source_(source)
        , first_(first)
        , rows_(rows)
        , width_(width) {}

    bool visit(const Def* value) {
        if (!depends(value)) return true;
        if (!seen_.insert(value).second) return true;
        if (++work_ > 262144) return false;
        if (value == source_) return false;
        if (auto region = input_region(value)) {
            RegionCoordinates coordinates(w_);
            if (!coordinates.collect(*region)) return false;
            for (auto point : coordinates.points)
                if (!visit(w_.app(w_.app(region->body(), point), region->values()))) return false;
            return true;
        }
        if (auto app = value->isa<App>();
            app && (Axm::isa<slice>(app) || Axm::isa<slice_box>(app)) && app->arg() == source_) {
            auto [meta, shape, params, source] = app->uncurry_args<4>();
            if (Lit::isa(meta->proj(2, 1)) != 2) return false;
            auto [off, middle, last] = params->projs<3>();
            auto origin              = lit_projs(off, 2);
            auto extent              = lit_projs(Axm::isa<slice>(app) ? last : middle, 2);
            if (Axm::isa<slice>(app)) {
                auto step = lit_projs(middle, 2);
                if (!step || (*step)[0] != 1 || (*step)[1] != 1) return false;
            }
            if (!origin || !extent || (*origin)[0] < first_ || (*origin)[0] - first_ > rows_
                || (*extent)[0] > rows_ - ((*origin)[0] - first_) || (*origin)[1] > width_
                || (*extent)[1] > width_ - (*origin)[1])
                return false;
            ++reads;
            return true;
        }
        if (value->isa<Var>() || value->isa<Axm>() || value->isa<Pi>()) return true;
        if (auto lam = value->isa<Lam>(); lam && lam->is_external()) return true;
        for (auto op : value->ops())
            if (op && !visit(op)) return false;
        return true;
    }
    u64 reads = 0;
    u64 work() const { return work_; }

private:
    bool depends(const Def* value) {
        if (value == source_) return true;
        if (auto known = dependencies_.find(value); known != dependencies_.end()) return known->second;
        dependencies_[value] = false;
        if (value->isa<Var>() || value->isa<Axm>() || value->isa<Pi>()) return false;
        for (auto op : value->ops())
            if (op && depends(op)) return dependencies_[value] = true;
        return false;
    }

    World& w_;
    const Def* source_;
    u64 first_, rows_, width_, work_ = 0;
    DefSet seen_;
    DefMap<bool> dependencies_;
};

} // namespace

static pattern::WitnessResult check_region_yield(const pattern::GraphView&, Defs, World&, bool = false);

pattern::WitnessResult outer_rows_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    using pattern::CheckResult;
    auto reject = [&](std::string reason) -> pattern::WitnessResult {
        target.log().d("tensor outer-rows candidate {}: reject {}", inputs[0], reason);
        return {CheckResult::False, nullptr, std::move(reason)};
    };
    auto region = input_region(inputs[0]);
    auto slot = Lit::isa(inputs[1]), tile = Lit::isa(inputs[2]);
    if (!region || !Axm::isa<tiled>(region->app) || !slot || !tile)
        return reject("row split requires a literal tiled policy");
    DefVec leaves;
    input_leaves(region->values(), leaves);
    auto producer = *slot < leaves.size() ? graph::map_reduce(leaves[*slot]) : nullptr;
    if (!producer) return reject("row split requires a direct MRP producer");
    bool outside = source.complete_data_uses(producer) != CheckResult::True;
    for (auto use : source.data_uses(producer))
        outside |= use.consumer != region->app;
    if (!outside || source.complete_data_uses(producer, true) != CheckResult::True)
        return reject("row split needs a proven outside reader");
    auto policy = row_policy(*region, *slot);
    if (!policy.reason.empty()) return reject(policy.reason);
    if (*tile != policy.tile) return reject("row split efficiency: tile differs from the cache-aligned policy");
    auto eligible = check_region_yield(
        source, DefVec{producer, region->app, region->body(), target.lit_idx(2, 0), inputs[1], target.lit_nat_1()},
        target, true);
    if (eligible.status != CheckResult::True) return reject(eligible.reason);
    auto occurrences = source.occurrences(region->app);
    if (occurrences.size() != 1 || !occurrences.front().regions.empty() || occurrences.front().trips != 1)
        return reject("row split needs one non-nested occurrence");
    auto scratch = target.inherit();
    pattern::ParameterImporter importer(*scratch);
    auto imported  = *input_region(importer.rewrite(region->app));
    auto p         = importer.rewrite(producer);
    auto plan      = grid_tile_plan(imported.app);
    auto shape     = *lit_projs(plan.shape, 2);
    auto width     = Lit::as(TensorType::isa(p->type())->s->proj(2, 1));
    auto outer     = *input_region(outer_rows(imported.app, *slot, *tile));
    u64 work       = 0;
    u64 proof_work = 0;
    auto limit     = mul(shape[0], width);
    if (!limit) return reject("row split producer work overflows");
    auto panel  = Lit::as(plan.tiles->proj(2, 1));
    auto column = scratch->mut_lam(scratch->type_nat(), scratch->type_nat())->var();
    DefVec widths{scratch->lit_nat(std::min(panel, shape[1]))};
    if (shape[1] % panel) widths.push_back(scratch->lit_nat(shape[1] % panel));
    for (u64 off = 0; off < shape[0]; off += std::min(*tile, shape[0] - off)) {
        auto ext = std::min(*tile, shape[0] - off);
        for (auto columns : widths) {
            auto value = scratch->app(scratch->app(imported.body(), {scratch->tuple({scratch->lit_nat(off), column}),
                                                                     scratch->tuple({scratch->lit_nat(ext), columns})}),
                                      imported.values());
            RowReadProof proof(*scratch, p, off, ext, width);
            if (!proof.visit(value) || !proof.reads)
                return reject("row split producer reads escape the owned whole-width rows or exceed proof limits");
            proof_work += proof.work();
            if (proof_work > 262144) return reject("row split exceeds the total read-proof limit");
        }
        auto coordinate = scratch->tuple({scratch->lit_nat(off), scratch->lit_nat(ext)});
        auto nested     = scratch->app(scratch->app(outer.body(), coordinate), outer.values());
        auto owned      = scratch->tuple({scratch->lit_nat(ext), scratch->lit_nat(width)});
        auto origin     = scratch->tuple({scratch->lit_nat(off), scratch->lit_nat_0()});
        auto tiled      = yield_tiles_value(p, origin, owned, nested, true, work);
        if (!tiled.value || work > *limit)
            return reject(tiled.value ? "row split exceeds one producer evaluation" : tiled.reason);
    }
    target.log().d("tensor outer-rows candidate {}: accept tile={} MR={} KC={} extra-weight-bytes={}", inputs[0], *tile,
                   policy.mr, policy.kc, policy.extra);
    return {CheckResult::True, target.lit_tt(), {}};
}

static pattern::WitnessResult
check_region_yield(const pattern::GraphView& source, Defs inputs, World& target, bool inputs_only) {
    using pattern::CheckResult;
    auto producer = graph::map_reduce(inputs[0]);
    auto region   = input_region(inputs[1]);
    auto slot = Lit::isa(inputs[4]), budget = Lit::isa(inputs[5]);
    if (!producer || !region || !slot || !budget || !*budget
        || (!Axm::isa<tiled>(region->app) && !Axm::isa<tiled_multi>(region->app)) || inputs[2] != region->body())
        return {CheckResult::False, nullptr, "yield requires a matching grid region and literal policy"};
    if (source.complete_data_uses(producer, true) != CheckResult::True)
        return {CheckResult::Unknown, nullptr, "yield requires complete producer data uses"};
    auto mr = btensor::MapRedPost::of(producer->callee()->as<App>());
    if (!pure_scalar(mr.comb) || !pure_scalar(mr.post))
        return {CheckResult::False, nullptr, "yield requires pure producer callbacks"};
    DefVec leaves;
    input_leaves(region->values(), leaves);
    if (*slot >= leaves.size() || leaves[*slot] != producer || std::ranges::count(leaves, producer) != 1)
        return {CheckResult::False, nullptr, "yield requires a single unaliased input position"};
    DefVec dependencies;
    input_leaves(producer->arg(), dependencies);
    for (size_t i = 0; i != leaves.size(); ++i)
        if (i != *slot && TensorType::isa(leaves[i]->type())
            && std::ranges::find(dependencies, leaves[i]) != dependencies.end())
            return {CheckResult::False, nullptr, "yield would alias another region input"};
    bool inside = false, outside = source.complete_data_uses(producer) != CheckResult::True;
    for (auto use : source.data_uses(producer)) {
        if (use.consumer == region->app) {
            if (use.group != region->input) return {CheckResult::False, nullptr, "yield has an aliased region input"};
            inside = true;
        } else
            outside = true;
    }
    if (!inside || !outside) return {CheckResult::False, nullptr, "yield needs both region and outside readers"};
    leaves[*slot]  = producer->arg();
    DefVec pending = leaves;
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i > 65536) return {CheckResult::Unknown, nullptr, "yield dependency proof exceeded its budget"};
        auto def = pending[i];
        if (def == producer) return {CheckResult::False, nullptr, "yield would create a cyclic region input"};
        if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>()) continue;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    auto occurrences = source.occurrences(region->app);
    if (occurrences.size() != 1 || !occurrences.front().regions.empty() || occurrences.front().trips != 1)
        return {CheckResult::Unknown, nullptr, "yield requires one non-nested grid occurrence"};
    if (inputs_only) return {CheckResult::True, target.lit_tt(), {}};
    auto scratch = target.inherit();
    pattern::ParameterImporter importer(*scratch);
    auto p        = importer.rewrite(producer);
    auto imported = input_region(importer.rewrite(region->app));
    auto plan     = grid_tile_plan(imported->app);
    auto axes     = importer.rewrite(inputs[3]);
    auto r        = Lit::isa(mr.Ro);
    if (!r) return {CheckResult::Unknown, nullptr, "yield needs a literal producer rank"};
    auto shape               = importer.rewrite(mr.So);
    std::optional<u64> whole = 1;
    for (u64 d = 0; d != *r; ++d)
        whole = mul(whole, Lit::isa(shape->proj(*r, d)));
    auto limit = mul(whole, budget);
    if (!limit) return {CheckResult::Unknown, nullptr, "yield has an unknown or overflowing work budget"};
    fe::Vector<bool> used(*r, false);
    for (u64 d = 0; d != plan.rank; ++d) {
        auto a = Lit::isa(axes->proj(plan.rank, d));
        if (!a || *a >= *r || used[*a] || shape->proj(*r, *a) != plan.shape->proj(plan.rank, d))
            return {CheckResult::False, nullptr, "yield result does not satisfy the grid box relation"};
        used[*a] = true;
    }
    GridTilePlan owned{importer.rewrite(mr.Tp),
                       importer.rewrite(mr.Ro),
                       shape,
                       plan.shape,
                       plan.tiles,
                       plan.tail,
                       axes,
                       plan.inputs,
                       plan.body,
                       1,
                       plan.rank,
                       false};
    RegionCoordinates coordinates(*scratch);
    if (!coordinates.collect(*imported))
        return {CheckResult::Unknown, nullptr, "yield needs bounded literal grid geometry"};
    u64 work = 0;
    DefSet extents;
    for (auto coordinate : coordinates.points) {
        auto [off, ext]      = coordinate->projs<2>();
        auto [start, extent] = owned.box(0, off, ext);
        auto value           = scratch->app(scratch->app(plan.body, coordinate), plan.inputs);
        auto result          = yield_tiles_value(p, start, extent, value, true, work);
        if (!result.value || work > *limit)
            return {CheckResult::False, nullptr,
                    result.value ? "yield exceeds the producer work budget" : result.reason};
        if (extents.insert(ext).second) {
            auto origin           = scratch->mut_lam(off->type(), scratch->type_nat())->var();
            auto [symbolic, size] = owned.box(0, origin, ext);
            auto body             = scratch->app(scratch->app(plan.body, {origin, ext}), plan.inputs);
            if (!yield_tiles_value(p, symbolic, size, body, false, work).value)
                return {CheckResult::False, nullptr, "yield cannot tile a runtime grid origin"};
        }
    }
    target.log().d("tensor fuse-and-yield candidate {}: work={} whole={} budget={}", producer, work, *whole, *budget);
    return {CheckResult::True, target.lit_tt(), {}};
}

pattern::WitnessResult region_yield_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    auto result = check_region_yield(source, inputs, target);
    if (result.status != pattern::CheckResult::True)
        target.log().d("tensor fuse-and-yield candidate {} region {}: reject {}", inputs[0], inputs[1], result.reason);
    return result;
}

namespace phase {

void FuseAndYield::apply(const App* app) {
    if (!Lit::isa(app->arg()) || !Lit::as(app->arg()))
        app->blame("tensor.fuse_and_yield requires a positive literal budget").bail();
    budget_ = Lit::as(app->arg());
}

std::unique_ptr<Phase> FuseAndYield::recreate() {
    auto phase     = std::make_unique<FuseAndYield>(world(), std::string(name()));
    phase->budget_ = budget_;
    return phase;
}

void FuseAndYield::start() {
    std::vector<const Rule*> rules;
    std::vector<const Axm*> pure;
    DefVec regions;
    {
        pattern::GraphView source(world(), driver().graph_operations());
        for (auto op : source.operations())
            if (Axm::isa<tiled>(op) || Axm::isa<tiled_multi>(op)) regions.push_back(op);
    }
    for (auto def : regions) {
        auto region = *input_region(def);
        auto plan   = grid_tile_plan(region.app);
        DefVec leaves;
        input_leaves(region.values(), leaves);
        for (size_t i = 0; i != leaves.size(); ++i)
            if (auto producer = graph::map_reduce(leaves[i])) {
                auto mr = btensor::MapRedPost::of(producer->callee()->as<App>());
                if (!Lit::isa(mr.Ro) || !mr.So->is_closed() || !plan.shape->is_closed()) {
                    log().d("tensor fuse-and-yield candidate {} input {}: reject unknown producer/grid geometry", def,
                            i);
                    continue;
                }
                YieldAxes axes(plan, mr);
                if (axes.choices.empty()) {
                    log().d("tensor fuse-and-yield candidate {} input {}: reject B1 grid shape {} producer shape {}",
                            def, i, plan.shape, mr.So);
                    if (driver().arg_nat("tensor", "outer-rows").value_or(1) && Axm::isa<tiled>(def) && plan.rank == 2
                        && Lit::isa(mr.Ro) == 2 && mr.So->proj(2, 0) == plan.shape->proj(2, 0)) {
                        auto policy = row_policy(region, i);
                        if (!policy.reason.empty())
                            log().d("tensor outer-rows candidate {} input {}: reject {}", def, i, policy.reason);
                        else
                            rules.push_back(outer_rows_rule(region, i, policy.tile));
                    }
                }
                for (auto mapping : axes.choices)
                    rules.push_back(yield_rule(region, i, mapping, budget_));
            } else {
                log().d("tensor fuse-and-yield candidate {} input {}: reject non-MRP input {}", def, i, leaves[i]);
            }
    }
    profile_count("tensor.yield.regions", regions.size());
    profile_count("tensor.yield.rules", rules.size());
    log().d("tensor fuse-and-yield candidates: regions={} rules={}", regions.size(), rules.size());
    if (rules.empty()) return;
    {
        pattern::GraphView source(world(), driver().graph_operations());
        for (auto node : source.nodes())
            if (auto axm = node->isa<Axm>())
                if (scalar_operator(axm) || axm->sym().view().starts_with("affine.")) pure.push_back(axm);
    }
    for (auto name :
         {"tensor.tiled", "tensor.tiled_multi", "tensor.map_reduce_post", "tensor.slice", "tensor.slice_box",
          "tensor.tile_of", "tensor.tile_of_box", "tensor.yield_tiles", "tensor.split_rows", "tensor.tile_fold",
          "tensor.splat", "tensor.generate", "tensor.transpose", "btensor.clip_sched", "core.idx", "math.F"})
        pure.push_back(world().annex(driver().sym(name))->as<Axm>());
    GraphRuleRewrite phase(world(), std::move(rules), std::move(pure), true);
    phase.run();
    invalidate(phase.todo());
    if (phase.todo()) profile_count("tensor.yield.rounds");
}

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
