#pragma once

#include <optional>

#include <fe/worklist.h>

#include <mim/lam.h>
#include <mim/tuple.h>
#include <mim/world.h>

#include "mim/plug/tensor/autogen.h"

namespace mim::plug::tensor {

/// Recognizes the (rebuilt) `tensor_copy` combiner `(acc, ys) ↦ ys#0`: the result is exactly the
/// single input element, so a map_reduce built on it is a pure re-indexed read of that input.
inline bool is_copy_comb(const Def* comb) {
    auto ret = Lam::isa_ret_arg(comb);
    return ret && ret == comb->as_mut<Lam>()->var(2, 0)->proj(2, 1)->proj(1, 0);
}

/// Is `post` the (rebuilt) CPS identity `tensor.id`, i.e. a lam `(x, extras) ↦ x` that returns its
/// first argument (and hence has no epilogue inputs)?
inline bool is_identity_post(const Def* post) {
    auto ret = Lam::isa_ret_arg(post);
    return ret && ret == post->as_mut<Lam>()->var(2, 0)->proj(2, 0);
}

/// A pure re-indexed read: the source tensor, the access map into it (over the read's output
/// coordinates), and the source's element type/rank/shape.
struct PureRead {
    const Def* src = nullptr;
    const Def* map = nullptr;
    const Def* T   = nullptr;
    const Def* R   = nullptr;
    const Def* S   = nullptr;
};

/// If `value` is a pure re-indexed read — a copy-combiner map_reduce without reduction loops that
/// writes its full loop domain through the identity output map (reshape/transpose/slice/flip/repeat
/// lower to these) — returns its single access map and source.
/// `fuse_tensor`'s read-through absorbs exactly these into the consuming op's access maps.
inline std::optional<PureRead> is_pure_read(const Def* value) {
    auto mr = Axm::isa<tensor::map_reduce_post>(value);
    if (!mr) return {};
    auto [nis_nps, meta, shapes, in_tys, comb_init, map_out, maps_all, is_all] = mr->uncurry_args<8>();

    auto [nis, nps] = nis_nps->projs<2>();
    // No reduction loops: the total loop count Rn equals the output rank Ro.
    if (Lit::isa(nis) != 1 || Lit::isa(nps) != 0 || meta->proj(5, 2) != meta->proj(5, 3)) return {};
    auto [So, Sr, sched] = shapes->projs<3>();
    if (Sr != So) return {};
    auto id_lam = map_out->isa_mut<Lam>();
    if (!id_lam || !id_lam->is_set() || id_lam->body() != id_lam->var()) return {};
    auto [comb, init, post] = comb_init->projs<3>();
    if (!is_copy_comb(comb) || !is_identity_post(post)) return {};

    auto sole = [](const Def* d, u64 n, u64 i) { return d->proj(n, i)->proj(1, 0); };
    return PureRead{sole(is_all, 2, 0), sole(maps_all, 2, 0), sole(in_tys, 6, 0), sole(in_tys, 6, 1),
                    sole(in_tys, 6, 2)};
}

/// The literal values of @p def's @p n projections, or nothing if one of them is not a literal.
inline std::optional<fe::Vector<u64>> lit_projs(const Def* def, u64 n) {
    auto res = fe::Vector<u64>(n);
    for (u64 i = 0; i != n; ++i)
        if (auto l = Lit::isa<u64>(def->proj(n, i)))
            res[i] = *l;
        else
            return {};
    return res;
}

/// @p perm as literals, if it is literal.
inline std::optional<fe::Vector<u64>> lit_perm(const Def* perm) {
    if (auto r = Lit::isa(perm->arity())) return lit_projs(perm, *r);
    return {};
}

/// The permutation of a `transpose` app, if it is literal.
inline std::optional<fe::Vector<u64>> transpose_perm(const App* app) {
    return lit_perm(app->decurry()->decurry()->arg());
}

/// The input of a 2-D transpose `transpose (1, 0)`, or `nullptr`.
inline const Def* isa_transpose_2d(const Def* def) {
    if (auto app = Axm::isa<tensor::transpose>(def)) {
        auto perm = app->decurry()->decurry()->arg();
        if (Lit::isa(perm->arity()) == 2) {
            auto [p0, p1] = perm->projs<2>();
            if (Lit::isa(p0) == 1 && Lit::isa(p1) == 0) return app->arg();
        }
    }
    return nullptr;
}

/// Is `map` the row-major reshape read `tensor.reshape_map (s_in, s_out)`, e.g. the one that reads a PACKED producer
/// (its output strip-mined to `s_in`) at the unpacked coordinates `s_out`?
/// Decided by normalization: both `map` and the canonical reshape map are applied to the same probe
/// variable; the reduced bodies are hash-consed, so pointer equality decides alpha-equivalence.
inline bool
is_reshape_read(World& w, const Def* map, const Def* r_in, const Def* s_in, const Def* r_out, const Def* s_out) {
    auto pi = map->type()->isa<Pi>();
    if (!pi) return false;
    auto expected = w.app(w.app(w.annex<tensor::reshape_map>(), {r_in, r_out}), {s_in, s_out});
    auto epi      = expected->type()->isa<Pi>();
    if (!epi || epi->dom() != pi->dom() || epi->codom() != pi->codom()) return false;
    auto probe = w.mut_lam(pi->dom(), pi->codom()); // scratch binder: its var stands in for the cell vector
    return w.app(map, probe->var()) == w.app(expected, probe->var());
}

/// Is `map` the row-major read of the leading `s_out#0` rows of `s_in`'s elements, e.g. the slice that drops a
/// register-blocked result's padding? Then the result is a prefix of the input buffer.
inline bool
is_prefix_read(World& w, const Def* map, const Def* r_in, const Def* s_in, const Def* r_out, const Def* s_out) {
    auto ri = Lit::isa<u64>(r_in), ro = Lit::isa<u64>(r_out);
    if (!ri || !ro || *ro == 0) return false;
    auto in = lit_projs(s_in, *ri), out = lit_projs(s_out, *ro);
    if (!in || !out) return false;
    u64 n = 1, row = 1;
    for (auto e : *in)
        n *= e;
    for (u64 d = 1; d != *ro; ++d)
        row *= (*out)[d];
    if (row == 0 || n % row != 0 || n / row < (*out)[0]) return false;
    auto pad = DefVec(*ro, [&](u64 d) { return d == 0 ? w.lit_nat(n / row) : s_out->proj(*ro, d); });
    return is_reshape_read(w, map, r_in, s_in, r_out, w.tuple(pad));
}

/// Counts the consumers of every def of @p world matched by @p pred.
/// Tuples and packs are transparent argument wrappers, so a wrapped def is charged to the enclosing
/// non-tuple consumer - a shared argument tuple charges each of its users, and a def used twice in one
/// argument list counts twice.
/// A phase whose world does not track uses needs this up front.
DefMap<u64> count_consumers(const World& world, std::predicate<const Def*> auto pred) {
    auto counts = DefMap<u64>();
    auto charge = [&](this auto&& charge, const Def* d) -> void {
        if (pred(d))
            ++counts[d];
        else if (d->isa<Tuple>() || d->isa<Pack>())
            for (auto op : d->ops())
                if (op) charge(op);
    };

    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(world.roots());

    while (!wl.empty()) {
        auto def         = wl.pop();
        auto transparent = def->isa<Tuple>() || def->isa<Pack>();
        for (auto op : def->ops())
            if (op) {
                if (!transparent) charge(op);
                wl.push(op);
            }
        if (def->type()) wl.push(def->type());
    }

    return counts;
}

} // namespace mim::plug::tensor
