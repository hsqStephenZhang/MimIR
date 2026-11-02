#include "mim/plug/tensor/pattern.h"

#include <limits>

#include <mim/pattern/match.h>

#include <mim/plug/core/core.h>

#include "mim/plug/tensor/autogen.h"

namespace mim::plug::tensor::graph {

const Def* slot(const Def* def, u64 count, u64 index) {
    if (!def || index >= count) return nullptr;
    if (count == 1) return def;
    if (auto tuple = def->isa<Tuple>(); tuple && tuple->num_ops() == count) return tuple->op(index);
    if (auto pack = def->isa_imm<Pack>(); pack && Lit::isa(*pack->shape()) == count) return pack->body();
    return nullptr;
}

pattern::Pattern call(pattern::PatternProgram& p, flags_t flags, std::vector<pattern::Pattern> groups) {
    auto result = p.node(Node::Axm, {}, flags);
    for (auto group : groups)
        result = p.node(Node::App, {result, group});
    return result;
}

namespace {

struct Calls {
    pattern::PatternProgram p;
    pattern::Pattern mr = call(p, Annex::base<map_reduce_post>(), std::vector(9, p.any()));
    pattern::Pattern bc = call(p, Annex::base<tensor::broadcast>(), std::vector(2, p.any()));
    pattern::Pattern mm = call(p, Annex::base<product_2d>(), std::vector(3, p.any()));

    const App* match(pattern::Pattern pat, const Def* def) const {
        return pattern::match(p, pat, def).status == pattern::MatchStatus::Matched ? def->as<App>() : nullptr;
    }
};

const Calls& calls() {
    static const Calls result;
    return result;
}

std::vector<const Def*> groups(const Def* def, size_t n) {
    std::vector<const Def*> result;
    for (auto cur = def; auto app = cur->isa<App>(); cur = app->callee()) {
        result.push_back(app->arg());
        if (result.size() > n) return {};
    }
    if (result.size() != n) return {};
    std::ranges::reverse(result);
    return result;
}

void inputs(pattern::OpView& view,
            const Def* values,
            u64 count,
            size_t group,
            std::vector<size_t> prefix,
            size_t limit) {
    if (count > limit - std::min(view.inputs.size(), limit)) {
        view.complete = false;
        return;
    }
    for (u64 i = 0; i != count; ++i) {
        auto value = slot(values, count, i);
        if (!value) {
            view.complete = false;
            continue;
        }
        auto path = prefix;
        path.push_back(i);
        view.inputs.push_back({value, group, std::move(path)});
    }
}

std::optional<pattern::OpView> decode_mr(const Def* def, size_t limit) {
    auto args = groups(def, 9);
    if (args.empty()) return {};
    pattern::OpView view;
    for (size_t group = 0; group != 2; ++group) {
        auto count = slot(args[0], 2, group);
        auto n     = count ? Lit::isa(count) : std::nullopt;
        if (!n) {
            view.complete = false;
            continue;
        }
        inputs(view, slot(args[8], 2, group), *n, 8, {group}, limit);
    }
    auto initial = slot(args[5], 3, 1);
    auto scalar  = slot(args[1], 5, 0);
    if (initial && scalar && !Check::test_alpha(initial->type(), scalar)) {
        auto count = initial->num_projs();
        if (auto value = (count == 2 || count == 3) ? slot(initial, count, count - 1) : nullptr)
            view.inputs.push_back({
                value,
                5,
                {1, count - 1}
            });
        else
            view.complete = false;
    }
    return view;
}

std::optional<pattern::OpView> decode_product(const Def* def, size_t limit) {
    auto args = groups(def, 3);
    if (args.empty()) return {};
    pattern::OpView view;
    inputs(view, args[2], 2, 2, {}, limit);
    return view;
}

std::optional<pattern::OpView> decode_concat(const Def* def, size_t limit) {
    auto args = groups(def, 4);
    if (args.empty()) return {};
    pattern::OpView view;
    auto count = slot(args[0], 3, 1);
    auto n     = count ? Lit::isa(count) : std::nullopt;
    if (n)
        inputs(view, args[3], *n, 3, {}, limit);
    else
        view.complete = false;
    return view;
}

std::optional<pattern::OpView> decode_broadcast(const Def* def, size_t limit) {
    auto args = groups(def, 2);
    if (args.empty()) return {};
    pattern::OpView view;
    auto input = slot(args[1], 3, 2);
    if (input && limit)
        view.inputs.push_back({input, 1, {2}});
    else
        view.complete = false;
    return view;
}

std::optional<pattern::OpView> decode_tiled(const Def* def, size_t) {
    auto args = groups(def, 3);
    if (args.empty()) return {};
    auto rank  = Lit::isa(slot(args[0], 2, 1));
    auto shape = slot(args[1], 2, 0), tile = slot(args[1], 2, 1);
    std::optional<u64> trips = 1;
    if (!rank || !shape || !tile)
        trips.reset();
    else
        for (u64 d = 0; d != *rank; ++d) {
            auto s = slot(shape, *rank, d), t = slot(tile, *rank, d);
            auto n = s ? Lit::isa(s) : std::nullopt, k = t ? Lit::isa(t) : std::nullopt;
            if (!n || !k || !*k) {
                trips.reset();
                break;
            }
            auto count = *n / *k + (*n % *k != 0);
            if (count && *trips > std::numeric_limits<u64>::max() / count) {
                trips.reset();
                break;
            }
            *trips *= count;
        }
    return pattern::OpView{.inputs = {}, .regions = {{args[2], trips}}};
}

std::optional<pattern::OpView> decode_fold(const Def* def, size_t) {
    auto args = groups(def, 4);
    if (args.empty()) return {};
    auto n = slot(args[1], 2, 0);
    return pattern::OpView{.inputs = {{args[2], 2, {}}}, .regions = {{args[3], n ? Lit::isa(n) : std::nullopt}}};
}

} // namespace

const App* map_reduce(const Def* def) { return calls().match(calls().mr, def); }
const App* broadcast(const Def* def) { return calls().match(calls().bc, def); }
const App* product(const Def* def) { return calls().match(calls().mm, def); }

bool is_standard_ring(const Def* ring) {
    auto type = slot(ring, 4, 0), zero = slot(ring, 4, 1);
    auto add = slot(ring, 4, 2), mul = slot(ring, 4, 3);
    if (!type || !zero || zero->type() != type || Lit::isa(zero) != 0 || !add || !mul) return false;
    if (type->node() == Node::Nat)
        return add->isa<Axm>() && add->flags() == flags_t(core::nat::add) && mul->isa<Axm>()
            && mul->flags() == flags_t(core::nat::mul);
    auto size = Idx::isa(type);
    if (!size || !Lit::isa(size)) return false;
    auto wraps = [&](const Def* f, core::wrap op) {
        auto args               = groups(f, 2);
        auto [axm, curry, trip] = Axm::get(f);
        return axm && axm->flags() == flags_t(op) && args.size() == 2 && Lit::isa(args[0]) == 0 && args[1] == size;
    };
    return wraps(add, core::wrap::add) && wraps(mul, core::wrap::mul);
}

void register_operations(pattern::OpRegistry& ops) {
    ops.add(Annex::base<map_reduce_post>(), decode_mr, false);
    ops.add(Annex::base<product_2d>(), decode_product,
            pattern::OpRegistry::Purity([](auto args) { return args.size() == 3 && is_standard_ring(args[0]); }));
    ops.add(Annex::base<tensor::broadcast>(), decode_broadcast, true);
    ops.add(Annex::base<concat>(), decode_concat, true);
    ops.add(Annex::base<slice>(), 4, {{3}}, true);
    ops.add(Annex::base<tiled>(), decode_tiled, false);
    ops.add(Annex::base<tile_fold>(), decode_fold, false);
    ops.add(Annex::base<stage>(), 5, {{4}}, true);
    ops.add(Annex::base<buf>(), 2, {{1}}, true);
}

} // namespace mim::plug::tensor::graph
