#include <optional>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/tuple.h>
#include <mim/world.h>

#include <mim/plug/core/core.h>

#include "mim/plug/affine/affine.h"

namespace mim::plug::affine {

namespace {

/// The values `[lo, hi]` an affine expression takes over a box, whether it takes `exact`ly all of them, and the
/// domain dims it depends on.
struct Itv {
    const Def* lo;
    const Def* hi;
    const Def* exact;
    u64 dims;
};

class Image {
public:
    Image(World& w, const Def* var, u64 n, const Def* lo, const Def* ext)
        : w_(w)
        , var_(var)
        , n_(n)
        , lo_(lo)
        , ext_(ext) {}

    std::optional<Itv> eval(const Def* e) {
        if (e == var_) return dim(0);
        if (auto ex = e->isa<Extract>(); ex && ex->tuple() == var_) {
            if (auto i = Lit::isa<u64>(ex->index()); i && *i < n_) return dim(*i);
            return {};
        }

        if (auto lit = Axm::isa<affine::lit>(e)) return Itv{lit->arg(), lit->arg(), w_.lit_tt(), 0};

        if (auto o = Axm::isa<affine::op>(e)) {
            if (o.id() == op::neg) return {};
            auto [a, b] = o->args<2>();
            auto x = eval(a), y = eval(b);
            if (!x || !y) return {};
            // A sum of two ranges is a range unless both depend on one dim (`i + i` skips the odd values).
            auto disjoint = (x->dims & y->dims) == 0 ? w_.lit_tt() : w_.lit_ff();
            switch (o.id()) {
                case op::add:
                    return Itv{add(x->lo, y->lo), add(x->hi, y->hi), all(x->exact, y->exact, disjoint),
                               x->dims | y->dims};
                case op::sub: {
                    auto point = y->dims == 0 ? y->exact : w_.lit_ff();
                    return Itv{sub(x->lo, y->hi), sub(x->hi, y->lo), all(x->exact, point, disjoint), x->dims | y->dims};
                }
                default: return {};
            }
        }

        if (auto o = Axm::isa<affine::semiop>(e)) {
            auto [a, c] = o->args<2>();
            auto x      = eval(a);
            if (!x) return {};
            switch (o.id()) {
                case semiop::mul: {
                    auto unit  = Lit::isa(c) == 1 || x->dims == 0 ? x->exact : w_.lit_ff();
                    auto exact = Lit::isa(c) == 0 ? w_.lit_tt() : unit;
                    return Itv{mul(x->lo, c), mul(x->hi, c), exact, x->dims};
                }
                case semiop::floordiv: return Itv{div(x->lo, c), div(x->hi, c), x->exact, x->dims};
                case semiop::ceildiv: return Itv{cdiv(x->lo, c), cdiv(x->hi, c), x->exact, x->dims};
                case semiop::rem: return rem(*x, c);
            }
        }

        if (auto l = Axm::isa<affine::linearize>(e)) return linearize(l);

        if (auto ex = e->isa<Extract>()) {
            if (auto d = Axm::isa<affine::delinearize>(ex->tuple()))
                if (auto k = Lit::isa<u64>(ex->index())) return delinearize(d, *k);
            return {};
        }
        if (auto d = Axm::isa<affine::delinearize>(e)) return delinearize(d, 0);

        return {};
    }

private:
    /// A dim of extent 1 is a single value, which any number of uses may share.
    Itv dim(u64 i) {
        auto lo  = lo_->proj(n_, i);
        auto ext = ext_->proj(n_, i);
        return Itv{lo, sub(add(lo, ext), w_.lit_nat_1()), w_.lit_tt(), Lit::isa(ext) == 1 ? 0 : u64(1) << i};
    }

    /// Without a carry across the box, `rem` is the shifted range; otherwise it is the whole `[0, c)`.
    Itv rem(const Itv& x, const Def* c) {
        auto same = w_.call(core::ncmp::e, Defs{div(x.lo, c), div(x.hi, c)});
        auto lo   = select(same, w_.call(core::nat::rem, Defs{x.lo, c}), w_.lit_nat_0());
        auto hi   = select(same, w_.call(core::nat::rem, Defs{x.hi, c}), sub(c, w_.lit_nat_1()));
        return Itv{lo, hi, all(x.exact, same, w_.lit_tt()), x.dims};
    }

    /// Row-major strides `∏_{j>k} s#j`; exact only while every index but the innermost is a single value.
    std::optional<Itv> linearize(const Def* app) {
        auto [idxs, s] = app->as<App>()->args<2>();
        auto r         = Lit::isa(s->arity());
        if (!r) return {};
        const Def* lo     = w_.lit_nat_0();
        const Def* hi     = w_.lit_nat_0();
        const Def* stride = w_.lit_nat_1();
        const Def* exact  = w_.lit_tt();
        u64 dims          = 0;
        for (u64 k = *r; k-- > 0;) {
            auto x = eval(idxs->proj(*r, k));
            if (!x) return {};
            lo    = add(lo, mul(x->lo, stride));
            hi    = add(hi, mul(x->hi, stride));
            exact = all(exact, x->exact, k + 1 == *r || x->dims == 0 ? w_.lit_tt() : w_.lit_ff());
            dims |= x->dims;
            stride = mul(stride, s->proj(*r, k));
        }
        return Itv{lo, hi, exact, dims};
    }

    /// Coordinate `k` of `delinearize (lin, s)` is `(lin floordiv ∏_{j>k} s#j) rem s#k`.
    std::optional<Itv> delinearize(const Def* app, u64 k) {
        auto [lin, s] = app->as<App>()->args<2>();
        auto r        = Lit::isa(s->arity());
        if (!r || k >= *r) return {};
        auto x = eval(lin);
        if (!x) return {};
        const Def* stride = w_.lit_nat_1();
        for (u64 j = k + 1; j < *r; ++j)
            stride = mul(stride, s->proj(*r, j));
        auto q = Itv{div(x->lo, stride), div(x->hi, stride), x->exact, x->dims};
        return rem(q, s->proj(*r, k));
    }

    const Def* add(const Def* a, const Def* b) { return w_.call(core::nat::add, Defs{a, b}); }
    const Def* sub(const Def* a, const Def* b) { return w_.call(core::nat::sub, Defs{a, b}); }
    const Def* mul(const Def* a, const Def* b) { return w_.call(core::nat::mul, Defs{a, b}); }
    const Def* div(const Def* a, const Def* b) { return w_.call(core::nat::div, Defs{a, b}); }
    const Def* cdiv(const Def* a, const Def* b) { return div(sub(add(a, b), w_.lit_nat_1()), b); }
    const Def* select(const Def* c, const Def* t, const Def* f) { return w_.extract(w_.tuple({f, t}), c); }
    const Def* all(const Def* a, const Def* b, const Def* c) {
        auto ab = w_.call(core::bit2::and_, w_.lit_nat(2), Defs{a, b});
        return w_.call(core::bit2::and_, w_.lit_nat(2), Defs{ab, c});
    }

    World& w_;
    const Def* var_;
    u64 n_;
    const Def* lo_;
    const Def* ext_;
};

} // namespace

const Def* normalize_image(const Def*, const Def* c, const Def* arg) {
    auto& w      = c->world();
    auto [nm, f] = c->as<App>()->uncurry_args<2>();
    auto [n, m]  = nm->projs<2>();
    auto ln = Lit::isa(n), lm = Lit::isa(m);
    if (!ln || !lm || *ln > 64) return nullptr;

    auto pi = f->type()->isa<Pi>();
    if (!pi) return nullptr;
    // Scratch binder: its var stands in for the domain point, as in `tensor`'s `is_unpack_read`.
    auto probe = w.mut_lam(pi->dom(), pi->codom());
    auto body  = w.app(f, probe->var());
    if (auto app = body->isa<App>(); app && app->callee() == f) return nullptr;

    auto [lo, ext, s] = arg->projs<3>();
    auto image        = Image(w, probe->var(), *ln, lo, ext);

    DefVec los(*lm), exts(*lm), exacts(*lm);
    for (u64 j = 0; j != *lm; ++j) {
        auto sj   = s->proj(*lm, j);
        auto last = w.call(core::nat::sub, Defs{sj, w.lit_nat_1()});
        auto itv  = image.eval(body->proj(*lm, j));
        if (!itv) itv = Itv{w.lit_nat_0(), last, w.lit_ff(), 0};
        auto in   = w.call(core::ncmp::l, Defs{itv->hi, sj});
        auto hi   = w.extract(w.tuple({last, itv->hi}), in);
        los[j]    = itv->lo;
        exts[j]   = w.call(core::nat::sub, Defs{w.call(core::nat::add, Defs{hi, w.lit_nat_1()}), itv->lo});
        exacts[j] = itv->exact;
    }

    return w.tuple({w.tuple(los), w.tuple(exts), w.tuple(exacts)});
}

MIM_affine_NORMALIZER_IMPL

} // namespace mim::plug::affine
