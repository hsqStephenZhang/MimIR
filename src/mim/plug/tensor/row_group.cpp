#include <mim/pattern/graph.h>

#include <mim/plug/affine/affine.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

namespace {

class RowRegion {
public:
    RowRegion(World& world, const Def* shape)
        : w(world)
        , shape(shape) {}

    bool collect(const Def* value, u64 rank = 0) {
        if (seen.contains(value)) return true;
        if (seen.size() == 64) return false;
        seen.insert(value);
        if (auto stage = Axm::isa<tensor::stage>(value)) {
            auto [meta, size, layout, scope] = stage->callee()->as<App>()->uncurry_args<4>();
            auto rank                        = Lit::isa(meta->proj(3, 1));
            if (!rank || !sizes(size, *rank) || Lit::isa(meta->proj(3, 2)) != 0) return false;
            for (size_t d = 0; d != *rank; ++d)
                if (Lit::isa(layout->proj(4, 2)->proj(*rank, d)) != d) return false;
            members.push_back(value);
            return collect(stage->arg(), *rank);
        }
        if (auto app = graph::map_reduce(value)) {
            auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
            auto ranks = mr.ranks();
            if (!ranks || ranks->ro < 1 || ranks->ro > 2 || ranks->nloops() < 1 || ranks->nloops() > 2
                || !sizes(mr.So, ranks->ro) || !sizes(mr.Sr, ranks->nloops()) || tensor_init(mr)
                || !pure_scalar(mr.comb) || !pure_scalar(mr.post)
                || !Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>()))
                return false;
            auto record = w.app(w.annex<tile_sched>(), mr.sched);
            if (!Check::test_alpha(mr.sched, w.app(w.annex<btensor::mk_sched>(), record))
                || !map(mr.map_out, ranks->nloops(), ranks->ro))
                return false;
            reduction |= ranks->rr != 0;
            members.push_back(value);
            auto [inputs, posts] = app->args<2>();
            for (auto [values, count, ranks_in, shapes, maps, domain] : {
                     std::tuple{inputs, mr.nis, mr.Ris, mr.Sis,      mr.maps, mr.Rn},
                     std::tuple{ posts, mr.nps, mr.Rps, mr.Sps, mr.post_maps, mr.Ro}
            }) {
                auto n = Lit::isa(count);
                if (!n) return false;
                for (u64 i = 0; i != *n; ++i) {
                    auto rank = Lit::isa(ranks_in->proj(*n, i));
                    if (!rank || !sizes(shapes->proj(*n, i), *rank) || !map(maps->proj(*n, i), Lit::as(domain), *rank)
                        || !collect(values->proj(*n, i), *rank))
                        return false;
                }
            }
            return true;
        }
        leaves[value] = rank;
        return rank && (value->isa<Var>() || Axm::isa<tensor::buf>(value));
    }

    const Def* build(const Def* value, const Def* off, const Def* ext) {
        if (auto i = local.find(value); i != local.end()) return i->second;
        auto size = [&](const Def* rank) { return Lit::as(rank) == 1 ? ext->proj(2, 0) : ext; };
        const Def* result;
        if (auto stage = Axm::isa<tensor::stage>(value)) {
            auto [meta, shape, layout, scope] = stage->callee()->as<App>()->uncurry_args<4>();
            result                            = w.app(
                w.app(w.app(w.app(w.app(w.annex<tensor::stage>(), meta), size(meta->proj(3, 1))), layout), scope),
                build(stage->arg(), off, ext));
        } else if (auto app = graph::map_reduce(value)) {
            auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
            auto slots = [&](const Def* values, const Def* count) {
                auto n = Lit::as(count);
                return w.tuple(DefVec(n, [&](size_t i) { return build(values->proj(n, i), off, ext); }));
            };
            auto shapes = [&](const Def* ranks, const Def* count) {
                auto n = Lit::as(count);
                return w.tuple(DefVec(n, [&](size_t i) { return size(ranks->proj(n, i)); }));
            };
            auto [inputs, posts] = app->args<2>();
            auto ins = w.tuple({mr.Tis, mr.Ris, shapes(mr.Ris, mr.nis), mr.Tps, mr.Rps, shapes(mr.Rps, mr.nps)});
            result   = w.annex<map_reduce_post>();
            for (auto arg : {mr.nis_nps, mr.meta, w.tuple({size(mr.Ro), size(mr.Rn), mr.sched}), ins, mr.init->type(),
                             w.tuple({mr.comb, mr.init, mr.post}), mr.map_out, mr.maps_all,
                             w.tuple({slots(inputs, mr.nis), slots(posts, mr.nps)})})
                result = w.app(result, arg);
        } else {
            auto rank         = leaves.at(value);
            auto element      = TensorType::isa(value->type())->T;
            auto source_shape = rank == 1 ? shape->proj(2, 0) : shape;
            auto slice        = w.app(w.annex<tensor::slice>(), {element, w.lit_nat(rank)});
            slice             = w.app(slice, source_shape);
            slice             = w.app(slice, {rank == 1 ? off->proj(2, 0) : off, w.pack(rank, w.lit_nat_1()),
                                              rank == 1 ? ext->proj(2, 0) : ext});
            result            = w.app(slice, value);
        }
        return local[value] = result;
    }

    World& w;
    const Def* shape;
    DefSet seen;
    DefMap<const Def*> local;
    DefMap<u64> leaves;
    DefVec members;
    bool reduction = false;

private:
    bool sizes(const Def* size, u64 rank) {
        return rank >= 1 && rank <= 2 && size == (rank == 1 ? shape->proj(2, 0) : shape);
    }

    bool map(const Def* fn, u64 domain, u64 range) {
        if (range > domain) return false;
        auto probe    = w.mut_lam(w.arr(domain, w.annex<affine::Index>()), w.type_nat());
        auto expected = w.tuple(DefVec(range, [&](size_t d) { return probe->var()->proj(domain, d); }));
        return w.app(fn, probe->var()) == expected;
    }
};

} // namespace

ProducerTile row_group(const Def* producer, DefVec& members) {
    auto app = graph::map_reduce(producer);
    if (!app) return {nullptr, "row region needs a map_reduce_post root"};
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = mr.ranks();
    auto shape = ranks ? lit_projs(mr.So, ranks->ro) : std::nullopt;
    if (!ranks || ranks->ro != 2 || !shape || (*shape)[0] < 2 || (*shape)[1] < 2)
        return {nullptr, "row region needs nontrivial literal row and column extents"};
    auto& w = producer->world();
    RowRegion region(w, mr.So);
    if (!region.collect(producer) || !region.reduction || region.members.size() < 3)
        return {nullptr, "row region needs a pure row-separable reduction closure"};
    auto tiled = w.app(w.annex<tensor::tiled>(), {mr.Tp, mr.Ro});
    tiled      = w.app(tiled, {mr.So, w.tuple({w.lit_nat_1(), mr.So->proj(2, 1)}), w.lit_nat_1()});
    RegionInputs inputs(region.members);
    auto values     = inputs.values(w);
    tiled           = w.app(w.app(tiled, values->type()), values);
    auto body       = w.mut_lam(tiled->type()->as<Pi>()->dom()->as<Pi>());
    auto [off, ext] = body->vars<2>();
    auto inner      = w.mut_lam(body->type()->reduce(body->var())->as<Pi>());
    inner->set(true, inputs.rebuild(region.build(producer, off, ext), inner->var()));
    body->set(true, inner);
    members = std::move(region.members);
    return {w.app(tiled, body), {}};
}

bool row_group_closed(const pattern::GraphView& source, const Def* producer) {
    auto scratch = producer->world().inherit();
    Rewriter importer(*scratch);
    DefVec members;
    if (!row_group(importer.rewrite(producer), members).value) return false;
    DefSet original;
    for (auto op : source.operations())
        if (std::ranges::find(members, importer.rewrite(op)) != members.end()) original.insert(op);
    if (original.size() != members.size()) return false;
    for (auto op : original) {
        if (op == producer) continue;
        if (source.complete_data_uses(op) != pattern::CheckResult::True) return false;
        for (auto use : source.data_uses(op))
            if (!original.contains(use.consumer)) return false;
    }
    return true;
}

} // namespace mim::plug::tensor
