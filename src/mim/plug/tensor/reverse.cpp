#include <mim/rewrite.h>

#include <mim/pattern/witness.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/core/core.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

namespace {

bool depends(const Def* value, Defs targets) {
    DefVec pending{value};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i == 16384) return true;
        auto def = pending[i];
        if (std::ranges::find(targets, def) != targets.end()) return true;
        if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return false;
}

DefVec consumers_of(const Def* value) {
    if (graph::map_reduce(value)) return {value};
    if (auto tuple = value->isa<Tuple>()) return DefVec(tuple->ops().begin(), tuple->ops().end());
    return {};
}

const Def*
region(flags_t axm, const Def* map, const Def* n, const Def* m, const Def* off, const Def* ext, const Def* shape) {
    auto& w = map->world();
    return w.app(w.app(w.app(w.annex(axm), {n, m}), map), {off, ext, shape});
}

const Def* zeros(World& w, u64 n) { return w.pack(n, w.lit_nat_0()); }

const Def* written_domain(const btensor::MapRedPost& op) {
    auto ranks = *op.ranks();
    auto& w    = op.Sr->world();
    return w.tuple(DefVec(ranks.nloops(), [&](size_t d) {
        auto size = op.Sr->proj(ranks.nloops(), d);
        return d >= ranks.ro && Lit::isa(size) == 0 ? w.lit_nat_1() : size;
    }));
}

bool exact_image(const Def* box, u64 rank) {
    auto& w = box->world();
    for (u64 d = 0; d != rank; ++d)
        if (box->proj(4, 3)->proj(rank, d) != w.lit_tt()) return false;
    return true;
}

std::string canonical(const btensor::MapRedPost& op) {
    auto& w    = op.Sr->world();
    auto ranks = op.ranks();
    if (!ranks || ranks->ro > 8 || ranks->nloops() > 64) return "ranks are not supported literals";
    if (!lit_projs(op.So, ranks->ro) || !lit_projs(op.Sr, ranks->nloops()))
        return "reverse placement requires literal shapes";
    if (!Check::test_alpha(op.TSched, w.annex<btensor::SchedT>()))
        return "custom schedule type does not prove surrounding loop order";
    auto record = w.app(w.annex<tile_sched>(), op.sched);
    if (!Check::test_alpha(op.sched, w.app(w.annex<btensor::mk_sched>(), record)))
        return "only canonical nests prove that just the placement prefix runs around the tile";
    if (!pure_scalar(op.comb) || !pure_scalar(op.post)) return "scalar callbacks are not proven pure";
    return {};
}

bool whole_write(const btensor::MapRedPost& op) {
    auto ranks = *op.ranks();
    auto& w    = op.Sr->world();
    auto box   = region(Annex::base<affine::image>(), op.map_out, op.Rn, op.Ro, zeros(w, ranks.nloops()),
                        written_domain(op), op.So);
    return exact_image(box, ranks.ro) && box->proj(4, 0) == zeros(w, ranks.ro) && box->proj(4, 1) == op.So;
}

struct ReverseGeometry {
    const Def* producer;
    DefVec consumers;
    DefVec ops;
    fe::Vector<DefVec> boxes;
    std::string reason;

    ReverseGeometry(const Def* p, const Def* cs, const Def* dim, const Def* tile_size)
        : producer(p)
        , consumers(consumers_of(cs)) {
        reason = plan(dim, tile_size);
    }

    std::string plan(const Def* dim, const Def* tile_size) {
        auto p = graph::map_reduce(producer);
        if (!p) return "producer is not a map_reduce_post";
        if (consumers.empty()) return "consumers must be a map_reduce_post or a nonempty tuple of them";
        auto mp = btensor::MapRedPost::of(p->callee()->as<App>());
        if (auto why = canonical(mp); !why.empty()) return "producer: " + why;
        auto pr   = *mp.ranks();
        auto axis = Lit::isa(dim), step = Lit::isa(tile_size);
        if (!axis || *axis >= pr.ro) return "placement dim is not a parallel producer dim";
        if (!step || !*step) return "tile size must be a positive literal";
        if (!whole_write(mp)) return "producer does not write its whole result exactly";
        ops.push_back(producer);
        DefVec maps;
        for (auto c : consumers) {
            auto app = graph::map_reduce(c);
            if (!app) return "consumer is not a map_reduce_post";
            if (std::ranges::find(ops, c) != ops.end()) return "producer and consumers must be distinct";
            auto mc = btensor::MapRedPost::of(app->callee()->as<App>());
            if (auto why = canonical(mc); !why.empty()) return "consumer: " + why;
            auto cr              = *mc.ranks();
            auto [inputs, posts] = app->args<2>();
            const Def* read      = nullptr;
            for (u64 j = 0; j != cr.nis; ++j) {
                auto input = inputs->proj(cr.nis, j);
                if (input == producer) {
                    if (read) return "consumer reads the producer through more than one input";
                    if (mc.Sis->proj(cr.nis, j) != mp.So) return "consumer input shape differs from producer output";
                    read = mc.maps->proj(cr.nis, j);
                } else if (depends(input, DefVec{producer}) || depends(input, consumers)) {
                    return "consumer side input depends on the producer or another selected consumer";
                }
            }
            if (!read) return "consumer does not directly read the selected producer";
            auto probe = p->world().mut_lam(read->type()->as<Pi>()->dom(), read->type()->as<Pi>()->codom());
            for (u64 j = 0; j != cr.nps; ++j) {
                auto input = posts->proj(cr.nps, j);
                if (input == producer) {
                    auto& w = p->world();
                    if (w.app(mc.post_maps->proj(cr.nps, j), w.app(mc.map_out, probe->var()))
                        != w.app(read, probe->var()))
                        return "consumer epilogue reads the producer outside its input tile";
                } else if (depends(input, DefVec{producer}) || depends(input, consumers)) {
                    return "consumer epilogue input depends on the producer or another selected consumer";
                }
            }
            if (auto init = tensor_init(mc);
                init && (depends(init->outs, DefVec{producer}) || depends(init->outs, consumers)))
                return "consumer accumulator depends on the producer or another selected consumer";
            if (depends(mc.comb, DefVec{producer}) || depends(mc.post, DefVec{producer}))
                return "consumer callback captures the producer";
            if (!whole_write(mc)) return "consumer does not write its whole result exactly";
            auto& w      = p->world();
            auto domain  = written_domain(mc);
            auto covered = region(Annex::base<affine::preimage>(), read, mc.Rn, mp.Ro, zeros(w, pr.ro), mp.So, domain);
            if (covered->proj(4, 3) != w.lit_tt() || covered->proj(4, 0) != zeros(w, cr.nloops())
                || covered->proj(4, 1) != domain)
                return "consumer reads outside the producer domain or its full read preimage is not exact";
            maps.push_back(read);
            ops.push_back(c);
        }
        auto& w     = p->world();
        auto domain = written_domain(mp);
        auto sizes  = *lit_projs(domain, pr.nloops());
        u64 trips   = 1;
        fe::Vector<u64> grid(*axis + 1);
        for (u64 d = 0; d <= *axis; ++d) {
            if (!sizes[d]) return "empty placement prefix is not supported";
            auto block = d == *axis ? *step : 1;
            grid[d]    = sizes[d] / block + (sizes[d] % block != 0);
            if (trips > 65536 / grid[d]) return "reverse placement exceeds 65536 grid points";
            trips *= grid[d];
        }
        for (u64 point = 0; point != trips; ++point) {
            DefVec origin(pr.nloops(), w.lit_nat_0()), extent(domain->projs(pr.nloops()));
            auto index = point;
            for (u64 d = *axis + 1; d-- != 0;) {
                auto start = index % grid[d] * (d == *axis ? *step : 1);
                index /= grid[d];
                origin[d] = w.lit_nat(start);
                extent[d] = w.lit_nat(std::min(d == *axis ? *step : 1, sizes[d] - start));
            }
            auto lo = w.tuple(origin), ext = w.tuple(extent);
            auto box = region(Annex::base<affine::image>(), mp.map_out, mp.Rn, mp.Ro, lo, ext, mp.So);
            if (!exact_image(box, pr.ro)) return "producer iteration write box is not exact";
            auto [off, len, cap, exact] = box->projs<4>();
            auto pre = region(Annex::base<affine::preimage>(), mp.map_out, mp.Rn, mp.Ro, off, len, domain);
            if (pre->proj(4, 3) != w.lit_tt()) return "producer write preimage is not exact";
            if (pre->proj(4, 0) != lo || pre->proj(4, 1) != ext)
                return "producer box is also written by another grid point";
            DefVec owned{w.tuple({off, len})};
            for (size_t j = 0; j != consumers.size(); ++j) {
                auto mc   = btensor::MapRedPost::of(consumers[j]->as<App>()->callee()->as<App>());
                auto cr   = *mc.ranks();
                auto cd   = written_domain(mc);
                auto part = region(Annex::base<affine::preimage>(), maps[j], mc.Rn, mp.Ro, off, len, cd);
                if (part->proj(4, 3) != w.lit_tt()) return "consumer tile read preimage is not exact";
                auto [start, size, bound, is_exact] = part->projs<4>();
                for (u64 d = cr.ro; d != cr.nloops(); ++d)
                    if (Lit::isa(start->proj(cr.nloops(), d)) != 0
                        || size->proj(cr.nloops(), d) != cd->proj(cr.nloops(), d))
                        return "consumer reduces across the producer tile; an ordered fold is required";
                auto output = region(Annex::base<affine::image>(), mc.map_out, mc.Rn, mc.Ro, start, size, mc.So);
                if (!exact_image(output, cr.ro)) return "consumer tile write image is not exact";
                auto co = output->proj(4, 0), ce = output->proj(4, 1);
                auto writers = region(Annex::base<affine::preimage>(), mc.map_out, mc.Rn, mc.Ro, co, ce, cd);
                if (writers->proj(4, 3) != w.lit_tt() || writers->proj(4, 0) != start || writers->proj(4, 1) != size)
                    return "consumer output box is not uniquely owned by this producer tile";
                owned.push_back(w.tuple({co, ce}));
            }
            boxes.push_back(std::move(owned));
        }
        return {};
    }
};

class TileReads : public Rewriter {
public:
    TileReads(World& w, const Def* producer, const Def* tile, const Def* off, const Def* ext)
        : Rewriter(w)
        , producer_(producer)
        , tile_(tile)
        , off_(off)
        , ext_(ext) {
        map(tile, tile);
    }

    const Def* rewrite_imm_App(const App* app) override {
        if (auto slice = Axm::isa<tensor::slice>(app); slice && slice->arg() == producer_) {
            auto [meta, shape, params, value] = slice->uncurry_args<4>();
            auto [off, step, ext]             = params->projs<3>();
            auto rank                         = Lit::as(meta->proj(2, 1));
            auto& w                           = world();
            auto start                        = w.tuple(DefVec(
                rank, [&](size_t d) { return w.call(core::nat::sub, Defs{off->proj(rank, d), off_->proj(rank, d)}); }));
            return w.app(w.app(w.app(w.app(w.annex<tensor::slice>(), meta), ext_), {start, step, ext}), tile_);
        }
        return Rewriter::rewrite_imm_App(app);
    }

private:
    const Def *producer_, *tile_, *off_, *ext_;
};

} // namespace

ProducerTile reverse_tiles(const Def* producer, const Def* consumers, const Def* dim, const Def* tile_size) {
    ReverseGeometry plan(producer, consumers, dim, tile_size);
    if (!plan.reason.empty()) return {nullptr, plan.reason};
    auto& w    = producer->world();
    auto count = plan.ops.size(), trips = plan.boxes.size();
    DefVec types, ranks, shapes, box_types;
    for (auto op : plan.ops) {
        auto mr = btensor::MapRedPost::of(op->as<App>()->callee()->as<App>());
        types.push_back(mr.Tp);
        ranks.push_back(mr.Ro);
        shapes.push_back(mr.So);
        box_types.push_back(w.app(w.annex<TileBox>(), mr.Ro));
    }
    auto boxes = w.mut_lam(w.type_nat(), w.sigma(box_types))->set("reverseBoxes");
    auto at    = w.call<core::idx>(w.lit_nat_0(), w.lit_nat(trips), boxes->var());
    boxes->set(true, w.tuple(DefVec(count, [&](size_t i) {
        return w.extract(w.tuple(DefVec(trips, [&](size_t t) { return plan.boxes[t][i]; })), at);
    })));
    auto result_types = [&](const Def* point) {
        auto owned = w.app(boxes, point);
        return w.sigma(DefVec(
            count, [&](size_t i) { return type_tensor(ranks[i], owned->proj(count, i)->proj(2, 1), types[i]); }));
    };
    auto pi = w.mut_pi(w.type())->set_dom(w.type_nat());
    RegionInputs inputs(plan.ops);
    auto values = inputs.values(w);
    pi->set_codom(w.pi(values->type(), result_types(pi->var())));
    auto body         = w.mut_lam(pi)->set("reverseTiles");
    auto inner        = w.mut_lam(body->type()->reduce(body->var())->as<Pi>());
    auto owned        = w.app(boxes, body->var());
    auto [poff, pext] = owned->proj(count, 0)->projs<2>();
    auto p            = w.app(w.app(w.annex<reverse_source>(), {types[0], ranks[0]}), shapes[0]);
    auto rebuilt      = inputs.rebuild(plan.ops, inner->var());
    p                 = w.app(p, {rebuilt.front(), poff, pext});
    DefVec tiles{p};
    for (size_t i = 1; i != count; ++i) {
        auto [off, ext] = owned->proj(count, i)->projs<2>();
        auto part       = w.app(w.annex<reverse_part>(),
                                {types[i], ranks[i], ranks[0], producer->type(), plan.ops[i]->type(), p->type()});
        tiles.push_back(w.app(part, {rebuilt[i], rebuilt.front(), p, off, ext, poff, pext}));
    }
    inner->set(true, w.tuple(tiles));
    body->set(true, inner);
    auto result = w.app(w.annex<tiled_n>(), {w.lit_nat(count), w.lit_nat_1(), w.tuple(types), w.tuple(ranks)});
    result      = w.app(
        w.app(w.app(w.app(w.app(result, {w.tuple(shapes), w.lit_nat(trips)}), boxes), values->type()), values), body);
    auto cs = graph::map_reduce(consumers)
                ? result->proj(count, 1)
                : w.tuple(DefVec(count - 1, [&](size_t i) { return result->proj(count, i + 1); }));
    return {w.tuple({result->proj(count, 0), cs}), {}};
}

const Def* normalize_reverse_build(const Def*, const Def*, const Def* arg) {
    auto [p, cs, dim, size] = arg->projs<4>();
    return reverse_tiles(p, cs, dim, size).value;
}

const Def* normalize_reverse_source(const Def*, const Def* callee, const Def* arg) {
    auto [producer, off, ext] = arg->projs<3>();
    auto rank                 = Lit::as(callee->as<App>()->callee()->as<App>()->arg(2, 1));
    if (!lit_projs(off, rank) || !lit_projs(ext, rank)) return nullptr;
    auto result = producer_tile(producer, off, ext);
    if (!result.value) producer->blame("reverse producer tile failed: {}", result.reason).bail();
    return result.value;
}

const Def* normalize_reverse_part(const Def*, const Def* callee, const Def* arg) {
    auto [consumer, producer, tile, off, ext, poff, pext] = arg->projs<7>();
    auto [type, rank, pr, ptype, ctype, tile_type]        = callee->as<App>()->args<6>();
    if (!lit_projs(off, Lit::as(rank)) || !lit_projs(ext, Lit::as(rank)) || !lit_projs(poff, Lit::as(pr))
        || !lit_projs(pext, Lit::as(pr)))
        return nullptr;
    auto result = producer_tile(consumer, off, ext);
    if (!result.value) consumer->blame("reverse consumer tile failed: {}", result.reason).bail();
    TileReads rewrite(consumer->world(), producer, tile, poff, pext);
    auto mr = btensor::MapRedPost::of(result.value->as<App>()->callee()->as<App>());
    rewrite.map(mr.comb, mr.comb);
    rewrite.map(mr.post, mr.post);
    for (auto var : result.value->free_vars())
        rewrite.map(var, var);
    return rewrite.rewrite(result.value);
}

pattern::WitnessResult reverse_candidate(const pattern::GraphView& source, Defs inputs, World& target) {
    using pattern::CheckResult;
    DefVec fields;
    for (u64 i = 0; i != 4; ++i) {
        auto value = graph::slot(inputs[0], 4, i);
        if (!value) return {CheckResult::Unknown, nullptr, "reverse placement needs frozen request fields"};
        fields.push_back(value);
    }
    auto consumers = consumers_of(fields[1]);
    if (consumers.empty()) return {CheckResult::False, nullptr, "reverse placement has no consumers"};
    for (auto c : consumers)
        if (source.data_used_by(fields[0], c) != CheckResult::True)
            return {CheckResult::Unknown, nullptr, "reverse placement needs a proven used_by edge"};
    auto scratch = target.inherit();
    Rewriter importer(*scratch);
    ReverseGeometry plan(importer.rewrite(fields[0]), importer.rewrite(fields[1]), importer.rewrite(fields[2]),
                         importer.rewrite(fields[3]));
    if (!plan.reason.empty()) return {CheckResult::False, nullptr, plan.reason};
    return {CheckResult::True, target.lit_tt(), {}};
}

} // namespace mim::plug::tensor
