#include <fe/bitset.h>

#include <mim/tuple.h>
#include <mim/world.h>

#include <mim/plug/core/core.h>

#include "mim/plug/vec/vec.h"

namespace mim::plug::vec {

/// How many elements @p vec holds, if that is statically known and small enough to unfold.
/// A Pack only unfolds for a @p type of `Nat`: anything wider is tensor data that has to stay a loop.
static std::optional<nat_t> isa_num_elems(const Def* vec, const Def* type) {
    if (auto tuple = vec->isa<Tuple>()) return tuple->num_ops();

    if (auto seq = vec->isa<Seq>()) {
        if (!seq->shape().is_fused())
            if (auto n = Lit::isa<u64>(seq->arity()); n && type->isa<Nat>()) return n;
        return {};
    }

    if (vec->isa<Lit>()) return 1; // `«1; E»` is `E`

    // A value that is neither a Tuple nor a Pack - a bufferized tensor, say - still unfolds into extracts,
    // but only while it is small enough to scalarize.
    auto n = std::optional<nat_t>();
    if (auto arr = vec->type()->isa<Arr>())
        n = arr->shape().is_fused() ? std::nullopt : Lit::isa<u64>(arr->arity());
    else if (auto sigma = vec->type()->isa<Sigma>())
        n = sigma->num_ops();

    return n && *n <= vec->world().flags().scalarize_threshold ? n : std::nullopt;
}

template<fold id>
const Def* normalize_fold(const Def* type, const Def* c, const Def* arg) {
    auto& w         = c->world();
    auto f          = c->as<App>()->arg();
    auto [acc, vec] = arg->projs<2>();
    if constexpr (id == fold::r) std::swap(acc, vec);

    auto n = isa_num_elems(vec, type);
    if (!n) return nullptr;

    if constexpr (id == fold::l)
        for (nat_t i = 0; i != *n; ++i)
            acc = w.app(f, {acc, vec->proj(*n, i)});
    else // fold::r
        for (nat_t i = *n; i-- != 0;)
            acc = w.app(f, {vec->proj(*n, i), acc});

    return acc;
}

const Def* normalize_zip(const Def* type, const Def* c, const Def* arg) {
    if (arg->is_open()) return {};
    auto& w           = type->world();
    auto [ni_n, _, f] = App::uncurry_args<3>(c);
    auto [ni, n]      = ni_n->projs<2>([](const Def* def) { return Lit::isa(def); });

    if (!ni || !n) return {};
    if (ni >= w.flags().scalarize_threshold || n >= w.flags().scalarize_threshold) return {};

    auto res = fe::Vector<const Def*, 32>(*n);
    auto tup = fe::Vector<const Def*, 32>(*ni);

    for (size_t j = 0; j != n; ++j) {
        for (size_t i = 0; i != ni; ++i)
            tup[i] = arg->proj(*ni, i)->proj(*n, j);

        res[j] = w.app(f, tup);
    }

    return w.tuple(res);
}

template<scan id>
const Def* normalize_scan(const Def*, const Def* c, const Def* vec) {
    auto& w     = c->world();
    auto callee = c->as<App>();
    auto p      = callee->arg();

    if (auto tuple = vec->isa<Tuple>()) {
        const Def* acc = w.lit_bool(id != scan::exists);
        for (auto op : tuple->ops())
            acc = w.call(id == scan::exists ? core::bit2::or_ : core::bit2::and_, 0_n, Defs{acc, w.app(p, op)});
        return acc;
    }

    if (auto pack = vec->isa_imm<Pack>()) w.log().w("Pack not yet implemented: {}", pack);

    return nullptr;
}

const Def* normalize_is_unique(const Def*, const Def*, const Def* vec) {
    auto& w = vec->world();

    if (auto tuple = vec->isa<Tuple>()) {
        auto seen = DefSet();
        for (auto op : tuple->ops()) {
            auto [_, ins] = seen.emplace(op);
            if (!ins) return w.lit_ff();
        }
        return tuple->is_closed() ? w.lit_tt() : nullptr;
    }

    if (auto pack = vec->isa_imm<Pack>()) {
        if (auto l = Lit::isa(pack->arity())) return w.lit_ff();
    }

    if (vec->isa<Lit>()) return w.lit_tt();

    return nullptr;
}

const Def* normalize_cat(const Def*, const Def* callee, const Def* arg) {
    auto [a, b] = arg->projs<2>();
    auto [n, m] = callee->as<App>()->decurry()->args<2>([](auto def) { return Lit::isa(def); });
    if (n && *n == 0) return b;
    if (m && *m == 0) return a;
    return n && m ? Tuple::cat(*n, *m, a, b) : nullptr;
}

const Def* normalize_diff(const Def* type, const Def* c, const Def* arg) {
    if (auto arr = type->isa<Arr>()) {
        if (arr->arity()->isa<Bot>()) return nullptr; // ack error
    }

    auto& w        = type->world();
    auto callee    = c->as<App>();
    auto [n, m]    = callee->args<2>([](auto def) { return Lit::isa(def); });
    auto [vec, is] = arg->projs<2>();

    if (!n || !m) return nullptr;
    if (n == 1 && m == 1) return w.tuple();

    if (auto tup_vec = vec->isa<Tuple>()) {
        if (auto tup_is = is->isa<Tuple>(); tup_is && tup_is->is_closed()) {
            auto defs = DefVec();
            auto drop = fe::Bitset();
            for (auto opi : tup_is->ops())
                drop.set(Lit::as(opi));

            for (size_t i = 0, e = tup_vec->num_ops(); i != e; ++i)
                if (!drop.test(i)) defs.emplace_back(tup_vec->op(i));
            return w.tuple(defs);
        }
        if (auto lit_is = Lit::isa(is)) {
            auto defs = DefVec();

            for (size_t i = 0, e = tup_vec->num_ops(); i != e; ++i)
                if (i != lit_is) defs.emplace_back(tup_vec->op(i));
            return w.tuple(defs);
        }
    }

    if (auto tup_pack = vec->isa_imm<Pack>()) return w.pack(*n - *m, tup_pack->body());

    return nullptr;
}

MIM_vec_NORMALIZER_IMPL

} // namespace mim::plug::vec
