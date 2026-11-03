#include <limits>

#include <mim/pattern/guard.h>
#include <mim/pattern/rule.h>
#include <mim/pattern/witness.h>

#include "mim/plug/affine/affine.h"
#include "mim/plug/core/core.h"
#include "mim/plug/tensor/phase/group_rules.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {
namespace {

class PackingImporter : public pattern::ParameterImporter {
public:
    using ParameterImporter::ParameterImporter;

private:
    const Def* rewrite_imm_Var(const Var* var) override {
        if (auto binder = lookup(var->binder())) return world().var(binder->as_mut());
        auto parameter = world().mut_lam(world().pi(rewrite(var->type()), world().type_nat()));
        parameter->set(true, world().lit_nat_0());
        if (var->binder()->is_external()) parameter->set("prepack_input")->externalize();
        return map(var, parameter->var());
    }
};

struct Packing {
    const Def* value = nullptr;
    std::string reason;
};

class Panel {
public:
    Panel(const App* app, Defs facts)
        : app_(app)
        , w_(app->world())
        , facts_(facts) {}

    Packing build(bool prove) {
        auto [meta, extent, layout, scope, input] = app_->uncurry_args<5>();
        auto [T, rank, split]                     = meta->projs<3>();
        if (Lit::isa(rank) != 2 || Lit::isa(split) != 1 || Lit::isa(scope) != 0)
            return {nullptr, "needs a two-dimensional scope-zero B panel"};
        auto [pos, tiles, permutation, pad] = layout->projs<4>();
        if (!pad->isa<Lit>() && !pad->isa<Bot>()) return {nullptr, "padding depends on another value"};
        auto tile = Lit::isa(tiles);
        auto perm = lit_projs(permutation, 2);
        if (Lit::isa(pos) != 1 || !tile || !*tile || !perm || *perm != fe::Vector<u64>{1, 0})
            return {nullptr, "needs the BLIS N/K/NR layout"};
        const Def* shape = extent;
        const Def* start = w_.tuple({w_.lit_nat_0(), w_.lit_nat_0()});
        input            = Axm::peel<buf>(input);
        if (auto slice = Axm::isa<tensor::slice>(input)) {
            auto [sm, ss, box, source] = slice->uncurry_args<4>();
            auto [off, step, size]     = box->projs<3>();
            auto steps                 = lit_projs(step, 2);
            if (!steps || *steps != fe::Vector<u64>{1, 1}) return {nullptr, "requires unit-stride panels"};
            start = off;
            shape = ss;
            input = Axm::peel<buf>(source);
        }
        auto parameter         = w_.mut_lam(w_.arr(2, w_.annex<affine::Index>()), w_.type_nat());
        auto variables         = parameter->var()->projs<2>();
        const Def* coordinates = parameter->var();
        while (!Axm::isa<frozen>(input)) {
            auto read = is_pure_read(input);
            if (!read) return {nullptr, "source is not a frozen whole input or a pure permutation of one"};
            coordinates = w_.app(read->map, coordinates);
            input       = Axm::peel<buf>(read->src);
        }
        auto marked = input;
        auto source = Axm::peel<buf>(input->as<App>()->arg());
        auto base   = source;
        while (auto ex = base->isa<Extract>())
            base = ex->tuple();
        auto var = base->isa<Var>();
        if (!var || !var->binder()->isa<Lam>() || !var->binder()->as<Lam>()->is_external())
            return {nullptr, "frozen source is not a whole function input"};
        auto type          = *TensorType::isa(source->type());
        auto rr            = Lit::isa(type.r);
        auto source_sizes  = rr ? lit_projs(type.s, *rr) : std::nullopt;
        auto logical_sizes = lit_projs(shape, 2);
        if (!source_sizes || !logical_sizes || *rr < 2) return {nullptr, "whole-input shapes must be literal"};
        fe::Vector<u64> axes(2, *rr), fixed;
        DefVec origin(*rr, w_.lit_nat_0());
        for (u64 d = 0; d != *rr; ++d) {
            auto coordinate = coordinates->proj(*rr, d);
            if (coordinate == variables[0])
                axes[0] = d;
            else if (coordinate == variables[1])
                axes[1] = d;
            else if (auto literal = Axm::isa<affine::lit>(coordinate)) {
                fixed.push_back(d);
                origin[d] = literal->arg();
                if (prove && bound(origin[d]) >= (*source_sizes)[d])
                    return {nullptr, "batch coordinate has no in-range literal bound"};
            } else
                return {nullptr, "source read is not a permutation with fixed batch coordinates"};
        }
        if (axes[0] == *rr || axes[1] == *rr || axes[0] == axes[1] || (*source_sizes)[axes[0]] != (*logical_sizes)[0]
            || (*source_sizes)[axes[1]] != (*logical_sizes)[1])
            return {nullptr, "source permutation changes matrix extents"};
        auto [ko, no] = start->projs<2>();
        auto [ke, ne] = extent->projs<2>();
        if (prove
            && (!multiple(no, *tile) || !panel_extent(no, ne, (*logical_sizes)[1], *tile)
                || !range(ko, ke, (*logical_sizes)[0])))
            return {nullptr, "panel alignment, tail padding or K range is unproved"};
        DefVec order;
        for (auto d : fixed)
            order.push_back(w_.lit_idx(*rr, d));
        order.push_back(w_.lit_idx(*rr, axes[1]));
        order.push_back(w_.lit_idx(*rr, axes[0]));
        auto packed_layout = w_.tuple({w_.lit_nat(axes[1]), tiles, w_.tuple(order), pad});
        auto packed        = w_.app(w_.app(w_.app(w_.annex<prepack>(), {T, type.r, split}), type.s), packed_layout);
        packed             = w_.app(packed, marked);
        auto packed_type   = *TensorType::isa(packed->type());
        auto sizes         = lit_projs(packed_type.s, *rr + 1);
        auto bytes         = Lit::isa(w_.app(w_.annex(core::trait::size), T));
        if (!sizes || !bytes) return {nullptr, "packed byte size is not literal"};
        u64 total = *bytes;
        for (auto size : *sizes) {
            if (size && total > std::numeric_limits<u64>::max() / size) return {nullptr, "packed byte size overflows"};
            total *= size;
        }
        if (total > w_.driver().arg_nat("tensor", "prepack-budget").value_or(64 * 1024 * 1024))
            return {nullptr, "packed copy exceeds the size budget"};
        DefVec offset, output;
        for (auto d : fixed) {
            offset.push_back(origin[d]);
            output.push_back(w_.lit_nat_1());
        }
        offset.insert(offset.end(), {w_.call(core::nat::div, Defs{no, tiles}), ko, w_.lit_nat_0()});
        auto panel_shape = TensorType::isa(app_->type())->s;
        auto panel_dims  = panel_shape->projs<3>();
        output.insert(output.end(), panel_dims.begin(), panel_dims.end());
        auto slice = w_.app(w_.app(w_.annex<tensor::slice>(), {T, w_.lit_nat(*rr + 1)}), packed_type.s);
        auto selected
            = w_.app(w_.app(slice, {w_.tuple(offset), w_.pack(*rr + 1, w_.lit_nat_1()), w_.tuple(output)}), packed);
        if (!fixed.empty()) {
            auto reshape = w_.app(w_.annex<tensor::reshape>(), {T, w_.lit_nat(*rr + 1), w_.lit_nat(3)});
            selected     = w_.app(w_.app(w_.app(reshape, w_.tuple(output)), panel_shape), selected);
        }
        return {selected, {}};
    }

private:
    std::optional<u64> fact(const Def* value, size_t field) {
        for (auto entry : facts_)
            if (entry->proj(3, 0) == value) return Lit::isa(entry->proj(3, field));
        return {};
    }
    u64 bound(const Def* value) {
        if (auto n = Lit::isa(value)) return *n;
        return fact(value, 2).value_or(std::numeric_limits<u64>::max());
    }
    bool multiple(const Def* value, u64 n) {
        if (auto x = Lit::isa(value)) return *x % n == 0;
        if (auto x = fact(value, 1)) return *x % n == 0;
        return false;
    }
    bool range(const Def* off, const Def* ext, u64 size) {
        auto a = bound(off), b = bound(ext);
        if (a <= size && b <= size - a) return true;
        for (auto entry : facts_)
            if (entry->proj(3, 0) == off && entry->proj(3, 1) == ext && Lit::isa(entry->proj(3, 2)) == size)
                return true;
        return false;
    }
    bool panel_extent(const Def* off, const Def* ext, u64 size, u64 tile) {
        if (!range(off, ext, size)) return false;
        if (multiple(ext, tile)) return true;
        auto a = Lit::isa(off), b = Lit::isa(ext);
        if (a && b && *a + *b == size) return true;
        for (auto entry : facts_)
            if (entry->proj(3, 0) == off && entry->proj(3, 1) == ext && Lit::isa(entry->proj(3, 2)) == size)
                return true;
        return false;
    }
    const App* app_;
    World& w_;
    Defs facts_;
};

} // namespace

pattern::WitnessResult prepack_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    auto scratch = source.world().inherit();
    PackingImporter importer(*scratch);
    auto candidate = importer.rewrite(inputs[0]);
    auto facts     = importer.rewrite(inputs[1]);
    auto stage     = Axm::isa<tensor::stage>(candidate);
    if (!stage) return {pattern::CheckResult::False, nullptr, "not a stage"};
    auto result = Panel(stage, facts->projs()).build(true);
    source.world().log().d("tensor prepack candidate {}: {}", inputs[0], result.value ? "accepted" : result.reason);
    return result.value ? pattern::WitnessResult{pattern::CheckResult::True, target.lit_tt(), {}}
                        : pattern::WitnessResult{pattern::CheckResult::False, nullptr, result.reason};
}

const Def* normalize_prepack_panel(const Def*, const Def*, const Def* arg) {
    if (auto stage = Axm::isa<tensor::stage>(arg)) return Panel(stage, {}).build(false).value;
    return nullptr;
}

namespace phase {
namespace {

struct Region {
    DefVec args;
    size_t input;
};

std::optional<Region> region_of(const Def* def) {
    auto app = def->isa<App>();
    if (!app) return {};
    if (Axm::isa<tiled>(app)) {
        auto a = app->uncurry_args<5>();
        return Region{
            {a.begin(), a.end()},
            3
        };
    }
    if (Axm::isa<tiled_multi>(app) || Axm::isa<tiled_n>(app) || Axm::isa<tile_fold>(app)) {
        auto a = app->uncurry_args<6>();
        return Region{
            {a.begin(), a.end()},
            4
        };
    }
    return {};
}

class PackRegions : public Rewriter {
public:
    using Rewriter::rewrite;
    using Rewriter::Rewriter;

    const Def* rewrite(const Def* def) override {
        if (auto known = lookup(def)) return known;
        if (auto var = def->isa<Var>()) {
            if (auto binder = lookup(var->binder())) return world().var(binder->as_mut());
            return def;
        }
        if (def->is_closed()) return def;
        return Rewriter::rewrite(def);
    }

private:
    const Def* rewrite_imm_Tuple(const Tuple* tuple) override { return world().tuple(Rewriter::rewrite(tuple->ops())); }

    bool packed_read(const Def* value) {
        value = Axm::peel<buf>(value);
        if (Axm::isa<prepack>(value)) return true;
        if (auto slice = Axm::isa<tensor::slice>(value)) return packed_read(slice->arg());
        if (auto read = is_pure_read(value)) return packed_read(read->src);
        if (auto reshape = Axm::isa<tensor::reshape>(value)) return packed_read(reshape->arg());
        return false;
    }

    void freeze_inputs(const Def* value) {
        if (auto tuple = value->isa<Tuple>())
            for (auto op : tuple->ops())
                freeze_inputs(op);
        else
            map(value, value);
    }

    void input_leaves(const Def* value, DefVec& inputs, DefSet& seen) {
        if (auto tuple = value->isa<Tuple>()) {
            for (auto op : tuple->ops())
                input_leaves(op, inputs, seen);
        } else if (packed_read(value)) {
            if (Axm::isa<prepack>(value)) {
                if (seen.insert(value).second) inputs.push_back(value);
            } else if (auto read = is_pure_read(value))
                input_leaves(read->src, inputs, seen);
            else if (auto app = value->isa<App>())
                input_leaves(app->arg(), inputs, seen);
        } else if (seen.insert(value).second)
            inputs.push_back(value);
    }

    void captures(const Def* value, const Var* coordinate, DefVec& inputs, DefSet& seen) {
        if (!seen.insert(value).second) return;
        if (Axm::isa<prepack>(value) && !value->has_free_var(coordinate)) {
            inputs.push_back(value);
            return;
        }
        if (value->isa<Var>() || value->isa<Axm>() || value->isa<Pi>()) return;
        for (auto op : value->ops())
            if (op) captures(op, coordinate, inputs, seen);
    }

    bool contains(const Def* value, const Def* needle, DefSet& seen) {
        if (value == needle) return true;
        if (!seen.insert(value).second || value->isa<Var>() || value->isa<Axm>() || value->isa<Pi>()) return false;
        for (auto op : value->ops())
            if (op && contains(op, needle, seen)) return true;
        return false;
    }

    const Def* rewrite_imm_App(const App* app) override {
        auto& w = world();
        if (auto region = region_of(app)) {
            auto args = region->args;
            auto body = args.back()->isa_mut<Lam>();
            if (!body || !body->is_set()) return Rewriter::rewrite_imm_App(app);
            auto before = changes_;
            auto inputs = rewrite(args[region->input]);
            freeze_inputs(inputs);
            auto saved      = facts_;
            auto coordinate = body->var();
            if (Axm::isa<tiled>(app)) {
                auto [shape, tiles] = args[1]->projs<2>();
                auto r              = Lit::isa(shape->arity());
                if (r)
                    for (u64 d = 0; d != *r; ++d) {
                        auto size = Lit::isa(shape->proj(*r, d)), tile = Lit::isa(tiles->proj(*r, d));
                        if (!size || !tile || !*tile) continue;
                        auto off = coordinate->proj(2, 0)->proj(*r, d);
                        auto ext = coordinate->proj(2, 1)->proj(*r, d);
                        facts_.push_back(w.tuple({off, w.lit_nat(*tile), w.lit_nat((*size - 1) / *tile * *tile)}));
                        facts_.push_back(
                            w.tuple({ext, w.lit_nat(*size % *tile ? 1 : *tile), w.lit_nat(std::min(*tile, *size))}));
                        facts_.push_back(w.tuple({off, ext, w.lit_nat(*size)}));
                    }
            } else if (Axm::isa<tile_fold>(app)) {
                auto [count, step] = args[1]->projs<2>();
                if (auto n = Lit::isa(count), k = Lit::isa(step); n && k && *n)
                    facts_.push_back(w.tuple({coordinate->proj(2, 1), step, w.lit_nat((*n - 1) * *k)}));
            }
            auto value = w.app(w.app(args.back(), coordinate), inputs);
            value      = rewrite(value);
            facts_     = std::move(saved);
            if (changes_ == before && inputs == args[region->input]) return app;
            DefVec data;
            DefSet seen;
            input_leaves(inputs, data, seen);
            std::erase_if(data, [&](const Def* input) {
                DefSet uses;
                return !contains(value, input, uses);
            });
            seen.clear();
            for (auto input : data)
                seen.insert(input);
            captures(value, body->has_var(), data, seen);
            auto captured      = w.tuple(data);
            const Def* builder = App::uncurry_callee(app);
            for (size_t i = 0; i + 1 != args.size(); ++i)
                builder = w.app(builder, i == region->input - 1 ? captured->type()
                                         : i == region->input   ? captured
                                                                : rewrite(args[i]));
            auto outer = w.mut_lam(builder->type()->as<Pi>()->dom()->as<Pi>());
            auto inner = w.mut_lam(outer->type()->reduce(outer->var())->as<Pi>());
            VarRewriter abstraction(w);
            for (auto var : value->free_vars())
                abstraction.add(var, var);
            abstraction.add(body->has_var(), outer->var());
            for (size_t i = 0; i != data.size(); ++i)
                abstraction.map(data[i], inner->var()->proj(data.size(), i));
            inner->set(true, abstraction.Rewriter::rewrite(value));
            outer->set(true, inner);
            return w.app(builder, outer);
        }
        auto result = Rewriter::rewrite_imm_App(app);
        if (auto stage = Axm::isa<tensor::stage>(result)) {
            auto meta      = stage->uncurry_args<5>()[0];
            auto [T, r, k] = meta->projs<3>();
            DefVec local;
            for (auto fact : facts_)
                if (result->has_free_vars_in(fact->free_vars())) local.push_back(fact);
            auto rule      = w.app(w.app(w.annex<prepack_rule>(), {T, r, k, w.lit_nat(local.size())}), w.tuple(local));
            auto templates = w.inherit();
            PackingImporter importer(*templates);
            auto candidate     = importer.rewrite(result);
            auto imported_rule = importer.rewrite(rule)->as<Rule>();
            Rewriter context(w);
            for (auto var : result->free_vars())
                context.map(importer.lookup(var), var);
            pattern::GraphView snapshot(Defs{candidate, importer.rewrite(w.tuple(local))});
            pattern::GuardContext guards(snapshot);
            auto replacement = pattern::RulePattern(imported_rule).build_replacement(candidate, context, {}, &guards);
            w.log().d("tensor prepack rule {}: {}", result, replacement.reason);
            if (replacement.status == pattern::MatchStatus::Budget_Exceeded)
                result->blame("tensor prepack rule exceeded its matching budget").bail();
            if (replacement.status == pattern::MatchStatus::Matched) {
                ++changes_;
                return replacement.value;
            }
        }
        return result;
    }

    DefVec facts_;
    size_t changes_ = 0;
};

} // namespace

void PrepackWeights::start() {
    bool frozen_input = false;
    {
        pattern::GraphView source(world(), driver().graph_operations());
        for (auto op : source.operations())
            frozen_input |= bool(Axm::isa<frozen>(op));
    }
    if (!frozen_input) return;
    DefVec roots;
    for (auto mut : world().externals().muts())
        if (auto lam = mut->isa<Lam>(); lam && lam->is_set()) roots.push_back(lam);
    for (auto root : roots) {
        auto lam = root->as_mut<Lam>();
        PackRegions rewrite(world());
        rewrite.map(lam->var(), lam->var());
        lam->set(lam->filter(), rewrite.rewrite(lam->body()));
    }
}

} // namespace phase
} // namespace mim::plug::tensor
