#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <ranges>
#include <vector>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/tuple.h>
#include <mim/world.h>

#include <mim/plug/core/core.h>

#include "mim/plug/affine/affine.h"

namespace mim::plug::affine {

namespace {

/// `Σ c#i · x#i + sym + k`, with literal coefficients; `sym` is a non-literal constant or `nullptr`.
struct Lin {
    std::vector<int64_t> c;
    const Def* sym = nullptr;
    int64_t k      = 0;

    bool constant() const {
        return std::ranges::all_of(c, [](int64_t ci) { return ci == 0; });
    }
    bool same_slope(const Lin& other) const { return c == other.c && sym == other.sym; }
};

/// The values `[lo, hi]` an affine expression takes over a box, a bound on their number that does not depend
/// on the box's origin, whether it takes `exact`ly all of them, the domain dims it depends on, and its linear
/// form if it has one.
/// A `clamped` bound may have saturated at 0 in `Nat` arithmetic, so it holds under the final clamp to the shape only.
struct Itv {
    const Def* lo;
    const Def* hi;
    const Def* width;
    const Def* exact;
    u64 dims;
    std::optional<Lin> lin;
    bool lo_clamped = false;
    bool hi_clamped = false;
};

class Region {
public:
    Region(World& w, const Def* var, u64 n, const Def* lo, const Def* ext)
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

        if (auto lit = Axm::isa<affine::lit>(e)) {
            auto l = Lin{std::vector<int64_t>(n_)};
            if (auto v = Lit::isa(lit->arg()))
                l.k = int64_t(*v);
            else
                l.sym = lit->arg();
            return Itv{lit->arg(), lit->arg(), one(), w_.lit_tt(), 0, l};
        }

        if (auto o = Axm::isa<affine::op>(e)) {
            if (o.id() == op::neg) return {};
            auto [a, b] = o->args<2>();
            auto x = eval(a), y = eval(b);
            if (!x || !y) return {};
            // A sum of two ranges is a range unless both depend on one dim (`i + i` skips the odd values).
            auto disjoint = (x->dims & y->dims) == 0 ? w_.lit_tt() : w_.lit_ff();
            auto width    = sub(add(x->width, y->width), one());
            auto dims     = x->dims | y->dims;
            switch (o.id()) {
                case op::add: {
                    auto exact = all(x->exact, y->exact, disjoint);
                    auto itv   = Itv{add(x->lo, y->lo), add(x->hi, y->hi), width, exact, dims, lin_add(x->lin, y->lin)};
                    return settle(itv, x->lo_clamped || y->lo_clamped, false, false);
                }
                case op::sub: {
                    auto exact = all(x->exact, y->dims == 0 ? y->exact : w_.lit_ff(), disjoint);
                    auto itv   = Itv{sub(x->lo, y->hi), sub(x->hi, y->lo), width, exact, dims, lin_sub(x->lin, y->lin)};
                    auto ge    = [](const Def* a, const Def* b) {
                        auto la = Lit::isa(a), lb = Lit::isa(b);
                        return lb == 0 || (la && lb && *la >= *lb);
                    };
                    return settle(itv, x->lo_clamped || y->lo_clamped, !ge(x->lo, y->hi), !ge(x->hi, y->lo));
                }
                default: return {};
            }
        }

        if (auto o = Axm::isa<affine::semiop>(e)) {
            auto [a, c] = o->args<2>();
            auto x      = eval(a);
            if (!x) return {};
            if (o.id() == semiop::mul) {
                auto unit  = Lit::isa(c) == 1 || x->dims == 0 ? x->exact : w_.lit_ff();
                auto exact = Lit::isa(c) == 0 ? w_.lit_tt() : unit;
                auto width = add(mul(sub(x->width, one()), c), one());
                auto itv   = Itv{mul(x->lo, c), mul(x->hi, c), width, exact, x->dims, lin_mul(x->lin, c)};
                return settle(itv, x->lo_clamped, false, false);
            }
            if (x->lo_clamped) return {};
            if (o.id() == semiop::floordiv || o.id() == semiop::rem)
                if (auto q = quotient(*x, c, o.id() == semiop::rem)) return q;
            switch (o.id()) {
                case semiop::mul: return {};
                case semiop::floordiv:
                    return Itv{div(x->lo, c), div(x->hi, c), div_width(*x, c, 0), x->exact, x->dims, {}};
                case semiop::ceildiv:
                    return Itv{cdiv(x->lo, c), cdiv(x->hi, c), div_width(*x, c, -1), x->exact, x->dims, {}};
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

    const Def* add(const Def* a, const Def* b) { return w_.call(core::nat::add, Defs{a, b}); }
    const Def* sub(const Def* a, const Def* b) { return w_.call(core::nat::sub, Defs{a, b}); }
    const Def* mul(const Def* a, const Def* b) { return w_.call(core::nat::mul, Defs{a, b}); }
    const Def* div(const Def* a, const Def* b) { return w_.call(core::nat::div, Defs{a, b}); }
    const Def* lt(const Def* a, const Def* b) { return w_.call(core::ncmp::l, Defs{a, b}); }
    const Def* select(const Def* c, const Def* t, const Def* f) { return w_.extract(w_.tuple({f, t}), c); }
    // No `Nat` is below 0.
    const Def* min(const Def* a, const Def* b) {
        if (a == w_.lit_nat_0() || b == w_.lit_nat_0()) return w_.lit_nat_0();
        return select(lt(a, b), a, b);
    }
    const Def* max(const Def* a, const Def* b) {
        if (a == w_.lit_nat_0()) return b;
        if (b == w_.lit_nat_0()) return a;
        return select(lt(a, b), b, a);
    }
    const Def* one() { return w_.lit_nat_1(); }
    const Def* all(const Def* a, const Def* b, const Def* c) {
        auto ab = w_.call(core::bit2::and_, w_.lit_nat(2), Defs{a, b});
        return w_.call(core::bit2::and_, w_.lit_nat(2), Defs{ab, c});
    }

private:
    /// A dim of extent 1 is a single value, which any number of uses may share.
    Itv dim(u64 i) {
        auto lo   = lo_->proj(n_, i);
        auto ex   = ext_->proj(n_, i);
        auto l    = Lin{std::vector<int64_t>(n_)};
        l.c[i]    = 1;
        auto unit = Lit::isa(ex) == 1;
        return Itv{lo, unit ? lo : sub(add(lo, ex), one()), ex, w_.lit_tt(), unit ? 0 : u64(1) << i, l};
    }

    /// Exact when every non-literal origin enters `x` through a multiple of `c`.
    std::optional<Itv> quotient(const Itv& x, const Def* c, bool rem) {
        auto n = Lit::isa<u64>(c);
        if (!n || *n == 0 || !x.lin || x.lin->sym) return {};
        auto b        = int64_t(*n);
        const Def* q  = w_.lit_nat_0();
        auto r_lo     = x.lin->k;
        auto r_hi     = x.lin->k;
        bool all_mult = x.lin->k % b == 0;
        for (u64 i = 0; i != n_; ++i) {
            auto ci = x.lin->c[i];
            if (ci == 0) continue;
            auto ext = Lit::isa<u64>(ext_->proj(n_, i));
            if (!ext) return {};
            auto span = ci * int64_t(*ext - 1);
            if (auto lo = Lit::isa<u64>(lo_->proj(n_, i))) {
                r_lo += ci * int64_t(*lo), r_hi += ci * int64_t(*lo) + span;
                all_mult &= ci % b == 0;
            } else {
                if (ci % b != 0) return {};
                q = add(q, mul(lo_->proj(n_, i), w_.lit_nat(u64(ci / b))));
                r_hi += span;
            }
        }
        if (r_lo < 0) return {};
        if (!rem) {
            auto lo  = add(q, w_.lit_nat(u64(r_lo / b)));
            auto hi  = add(q, w_.lit_nat(u64(r_hi / b)));
            auto lin = std::optional<Lin>();
            if (all_mult) {
                lin = *x.lin;
                for (auto& ci : lin->c)
                    ci /= b;
                lin->k /= b;
            }
            return Itv{lo, hi, w_.lit_nat(u64(r_hi / b - r_lo / b + 1)), x.exact, x.dims, lin};
        }
        // A contiguous run of at least `c` values takes every residue.
        if (r_lo / b == r_hi / b) {
            auto lo = w_.lit_nat(u64(r_lo % b)), hi = w_.lit_nat(u64(r_hi % b));
            return Itv{lo, hi, w_.lit_nat(u64(r_hi - r_lo + 1)), x.exact, x.dims, {}};
        }
        auto whole = r_hi - r_lo + 1 >= b ? x.exact : w_.lit_ff();
        return Itv{w_.lit_nat_0(), w_.lit_nat(u64(b - 1)), c, whole, x.dims, {}};
    }

    const Def* cdiv(const Def* a, const Def* b) { return div(sub(add(a, b), one()), b); }

    /// Bounds built from @p operands that may have saturated are recomputed from the linear form, which saturates
    /// only once at its end, or dropped; @p lo and @p hi state that this op itself may have saturated them.
    std::optional<Itv> settle(Itv itv, bool operands, bool lo, bool hi) {
        if (!operands && !lo && !hi) return itv;
        if (itv.lin) return from_lin(itv);
        if (operands) return {};
        itv.lo_clamped = lo || hi;
        itv.hi_clamped = hi;
        return itv;
    }

    /// A non-literal origin or extent is at least 0 or 1, which bounds whether its sum may drop below 0.
    Itv from_lin(Itv itv) {
        auto& l        = *itv.lin;
        const Def* lo  = l.sym ? l.sym : w_.lit_nat_0();
        const Def* hi  = lo;
        int64_t lo_min = 0, hi_min = 0;
        for (u64 i = 0; i != n_; ++i) {
            if (l.c[i] == 0) continue;
            auto c  = w_.lit_nat(u64(l.c[i]));
            auto li = lo_->proj(n_, i);
            auto ei = ext_->proj(n_, i);
            lo      = add(lo, mul(li, c));
            hi      = add(hi, mul(sub(add(li, ei), one()), c));
            lo_min += l.c[i] * int64_t(Lit::isa(li).value_or(0));
            hi_min += l.c[i] * int64_t(Lit::isa(li).value_or(0) + Lit::isa(ei).value_or(1) - 1);
        }
        auto k         = w_.lit_nat(u64(l.k < 0 ? -l.k : l.k));
        itv.lo         = l.k < 0 ? sub(lo, k) : add(lo, k);
        itv.hi         = l.k < 0 ? sub(hi, k) : add(hi, k);
        itv.lo_clamped = lo_min + l.k < 0;
        itv.hi_clamped = hi_min + l.k < 0;
        return itv;
    }

    /// `w` consecutive values from `v` take `(v rem c + w − 1)/c + 1` quotients by `c`, and never more than `w`.
    /// `v rem c` is bounded by the residues `v` admits, as a non-literal origin may be any value.
    const Def* div_width(const Itv& x, const Def* c, int64_t shift) {
        auto loose = min(add(div(sub(x.width, one()), c), w_.lit_nat(2)), x.width);
        auto n     = Lit::isa(c);
        if (!n || *n == 0 || !x.lin) return loose;
        auto g = *n;
        auto k = x.lin->k + shift;
        if (x.lin->sym) g = 1;
        for (u64 i = 0; i != n_; ++i) {
            if (x.lin->c[i] == 0) continue;
            if (auto lo = Lit::isa(lo_->proj(n_, i)))
                k += x.lin->c[i] * int64_t(*lo);
            else
                g = std::gcd(g, u64(x.lin->c[i]));
        }
        auto rho = *n - g + u64((k % int64_t(g) + int64_t(g)) % int64_t(g));
        return min(add(div(add(w_.lit_nat(rho), sub(x.width, one())), c), one()), loose);
    }

    /// Without a carry across the box, `rem` is the shifted range; otherwise it is the whole `[0, c)`.
    /// Whether it carries depends on the origin, so only a single value is bounded by less than `c`.
    Itv rem(const Itv& x, const Def* c) {
        auto same  = w_.call(core::ncmp::e, Defs{div(x.lo, c), div(x.hi, c)});
        auto lo    = select(same, w_.call(core::nat::rem, Defs{x.lo, c}), w_.lit_nat_0());
        auto hi    = select(same, w_.call(core::nat::rem, Defs{x.hi, c}), sub(c, one()));
        auto width = select(w_.call(core::ncmp::e, Defs{x.width, one()}), one(), c);
        return Itv{lo, hi, width, all(x.exact, same, w_.lit_tt()), x.dims, {}};
    }

    /// Row-major strides `∏_{j>k} s#j`; exact only while every index but the innermost is a single value.
    std::optional<Itv> linearize(const Def* app) {
        auto [idxs, s] = app->as<App>()->args<2>();
        auto r         = Lit::isa(s->arity());
        if (!r) return {};
        const Def* lo     = w_.lit_nat_0();
        const Def* hi     = w_.lit_nat_0();
        const Def* width  = one();
        const Def* stride = one();
        const Def* exact  = w_.lit_tt();
        u64 dims          = 0;
        auto lin          = std::optional<Lin>(Lin{std::vector<int64_t>(n_)});
        bool clamped      = false;
        for (u64 k = *r; k-- > 0;) {
            auto x = eval(idxs->proj(*r, k));
            if (!x) return {};
            clamped |= x->lo_clamped;
            lo    = add(lo, mul(x->lo, stride));
            hi    = add(hi, mul(x->hi, stride));
            width = add(width, mul(sub(x->width, one()), stride));
            exact = all(exact, x->exact, k + 1 == *r || x->dims == 0 ? w_.lit_tt() : w_.lit_ff());
            dims |= x->dims;
            lin    = lin_add(lin, lin_mul(x->lin, stride));
            stride = mul(stride, s->proj(*r, k));
        }
        return settle(Itv{lo, hi, width, exact, dims, lin}, clamped, false, false);
    }

    /// Coordinate `k` of `delinearize (lin, s)` is `(lin floordiv ∏_{j>k} s#j) rem s#k`.
    std::optional<Itv> delinearize(const Def* app, u64 k) {
        auto [lin, s] = app->as<App>()->args<2>();
        auto r        = Lit::isa(s->arity());
        if (!r || k >= *r) return {};
        auto x = eval(lin);
        if (!x || x->lo_clamped) return {};
        const Def* stride = one();
        for (u64 j = k + 1; j < *r; ++j)
            stride = mul(stride, s->proj(*r, j));
        auto q = Itv{div(x->lo, stride), div(x->hi, stride), div_width(*x, stride, 0), x->exact, x->dims, {}};
        return rem(q, s->proj(*r, k));
    }

    std::optional<Lin> lin_add(const std::optional<Lin>& a, const std::optional<Lin>& b) {
        if (!a || !b || (a->sym && b->sym)) return {};
        auto l = *a;
        for (u64 i = 0; i != n_; ++i)
            l.c[i] += b->c[i];
        l.sym = a->sym ? a->sym : b->sym;
        l.k += b->k;
        return l;
    }

    /// Only a constant is subtracted, so the coefficients stay non-negative.
    std::optional<Lin> lin_sub(const std::optional<Lin>& a, const std::optional<Lin>& b) {
        if (!a || !b || !b->constant() || (b->sym && b->sym != a->sym)) return {};
        auto l = *a;
        if (b->sym) l.sym = nullptr;
        l.k -= b->k;
        return l;
    }

    std::optional<Lin> lin_mul(const std::optional<Lin>& a, const Def* c) {
        auto v = Lit::isa(c);
        if (!a || !v || (a->sym && *v != 1)) return {};
        auto l = *a;
        for (auto& ci : l.c)
            ci *= int64_t(*v);
        l.k *= int64_t(*v);
        return l;
    }

    World& w_;
    const Def* var_;
    u64 n_;
    const Def* lo_;
    const Def* ext_;
};

/// The body of `f` at a scratch binder's var, as `tensor`'s `is_unpack_read` probes a map; `nullptr` while `f`
/// does not β-reduce.
std::pair<const Def*, const Def*> probe(const Def* f) {
    auto& w = f->world();
    auto pi = f->type()->isa<Pi>();
    if (!pi) return {};
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto body = w.app(f, var);
    if (auto app = body->isa<App>(); app && app->callee() == f) return {};
    return {var, body};
}

/// The hull of the coordinate `j` of all `maps` over the box, clamped to `[0, s)`: `(lo, ext, cap, exact)`.
std::array<const Def*, 4> hull(Region& r, const DefVec& bodies, u64 m, u64 j, const Def* s) {
    auto& w   = s->world();
    auto last = r.sub(s, r.one());

    auto itvs = std::vector<Itv>();
    for (auto body : bodies)
        if (auto itv = r.eval(body->proj(m, j)))
            itvs.emplace_back(*itv);
        else
            return {w.lit_nat_0(), s, s, w.lit_ff()};

    const Def *lo, *hi, *width, *exact;
    auto offsets = [&]() {
        for (auto& itv : itvs)
            if (!itv.lin || !itv.lin->same_slope(*itvs.front().lin) || !Lit::isa(itv.width)) return false;
        return true;
    };

    if (itvs.size() == 1) {
        lo = itvs[0].lo, hi = itvs[0].hi, width = itvs[0].width, exact = itvs[0].exact;
    } else if (offsets()) {
        // The maps are one map shifted by the literal offsets `k`, so the hull is too.
        auto first = std::ranges::min_element(itvs, {}, [](const Itv& itv) { return itv.lin->k; });
        auto reach = std::ranges::max(
            itvs | std::views::transform([](const Itv& itv) { return itv.lin->k + int64_t(*Lit::isa(itv.width)); }));
        auto size = u64(reach - first->lin->k);
        // The union of ranges is a range if, in order of their starts, each one starts within its predecessors.
        std::ranges::sort(itvs, {}, [](const Itv& itv) { return itv.lin->k; });
        exact = w.lit_tt();
        for (int64_t end = itvs.front().lin->k; auto& itv : itvs) {
            exact = r.all(exact, itv.exact, itv.lin->k <= end ? w.lit_tt() : w.lit_ff());
            end   = std::max(end, itv.lin->k + int64_t(*Lit::isa(itv.width)));
        }
        lo    = itvs.front().lo;
        hi    = itvs.back().hi;
        width = w.lit_nat(size);
    } else {
        lo = itvs[0].lo, hi = itvs[0].hi;
        for (auto& itv : itvs)
            lo = r.min(lo, itv.lo), hi = r.max(hi, itv.hi);
        width = s, exact = w.lit_ff();
    }

    // An upper bound that saturated claims 0, which the box need not reach.
    if (std::ranges::any_of(itvs, [](const Itv& itv) { return itv.hi_clamped; })) exact = w.lit_ff();
    auto top = r.min(hi, last);
    return {lo, r.sub(r.add(top, r.one()), lo), r.min(width, s), exact};
}

const Def* empty_region(World& w, u64 m) {
    auto zeros = w.tuple(DefVec(m, [&](size_t) -> const Def* { return w.lit_nat_0(); }));
    return w.tuple({zeros, zeros, zeros, w.tuple(DefVec(m, [&](size_t) -> const Def* { return w.lit_tt(); }))});
}

const Def* boxes(const Def* fs, u64 k, u64 n, u64 m, const Def* box) {
    auto& w = fs->world();
    if (k == 0) return empty_region(w, m);

    const Def* var = nullptr;
    auto bodies    = DefVec(k);
    for (u64 i = 0; i != k; ++i) {
        auto f         = fs->proj(k, i);
        auto [v, body] = probe(f);
        if (!body) return nullptr;
        // Every map gets its own scratch binder; rename them all to the first one's var.
        if (var)
            body = w.app(f, var);
        else
            var = v;
        bodies[i] = body;
    }

    auto [lo, ext, s] = box->projs<3>();
    auto region       = Region(w, var, n, lo, ext);
    // An empty domain reaches nothing, while the closed bounds `lo + ext − 1` would saturate to a value.
    const Def* empty = w.lit_ff();
    for (u64 i = 0; i != n; ++i)
        empty = w.call(core::bit2::or_, w.lit_nat(2),
                       Defs{empty, w.call(core::ncmp::e, Defs{ext->proj(n, i), w.lit_nat_0()})});
    if (empty == w.lit_tt()) return empty_region(w, m);

    auto res = std::array<DefVec, 4>{DefVec(m), DefVec(m), DefVec(m), DefVec(m)};
    for (u64 j = 0; j != m; ++j) {
        auto h = hull(region, bodies, m, j, s->proj(m, j));
        for (size_t x = 0; x != 4; ++x)
            res[x][j] = x == 2 ? h[x] : region.select(empty, x == 3 ? w.lit_tt() : w.lit_nat_0(), h[x]);
    }
    return w.tuple({w.tuple(res[0]), w.tuple(res[1]), w.tuple(res[2]), w.tuple(res[3])});
}

} // namespace

const Def* normalize_image(const Def*, const Def* c, const Def* arg) {
    auto [nm, f] = c->as<App>()->uncurry_args<2>();
    auto [n, m]  = nm->projs<2>();
    auto ln = Lit::isa(n), lm = Lit::isa(m);
    if (!ln || !lm || *ln > 64) return nullptr;
    return boxes(f, 1, *ln, *lm, arg);
}

const Def* normalize_cover(const Def*, const Def* c, const Def* arg) {
    auto [knm, fs] = c->as<App>()->uncurry_args<2>();
    auto [k, n, m] = knm->projs<3>();
    auto lk = Lit::isa(k), ln = Lit::isa(n), lm = Lit::isa(m);
    if (!lk || !ln || !lm || *ln > 64) return nullptr;
    return boxes(fs, *lk, *ln, *lm, arg);
}

const Def* normalize_preimage(const Def*, const Def* c, const Def* arg) {
    auto& w      = c->world();
    auto [nm, f] = c->as<App>()->uncurry_args<2>();
    auto [n, m]  = nm->projs<2>();
    auto ln = Lit::isa(n), lm = Lit::isa(m);
    if (!ln || !lm || *ln > 64) return nullptr;

    auto [var, body] = probe(f);
    if (!body) return nullptr;

    auto [lo, ext, S] = arg->projs<3>();
    auto dom_lo       = w.tuple(DefVec(*ln, [&](size_t) -> const Def* { return w.lit_nat_0(); }));
    auto region       = Region(w, var, *ln, dom_lo, S);

    auto los   = DefVec(*ln, [&](size_t) -> const Def* { return w.lit_nat_0(); });
    auto exts  = DefVec(*ln, [&](size_t i) { return S->proj(*ln, i); });
    auto caps  = exts;
    auto seen  = std::vector<bool>(*ln);
    bool exact = true;
    auto one   = [](const Lin& l) {
        auto nz = std::ranges::count_if(l.c, [](int64_t ci) { return ci != 0; });
        return nz == 1 && std::ranges::count(l.c, 1) == 1;
    };
    // A dim's quotient and remainder coordinates only bound it together.
    struct Split {
        u64 b         = 0;
        const Def *lo = nullptr, *ext = nullptr;
    };
    auto quot = std::vector<Split>(*ln), rems = std::vector<Split>(*ln);
    // `q·x#a + x#b` with `x#b` spanning `[0, q)`, as a split dim writes: a box of one cell, or of whole runs of `q`
    // from a multiple of `q`, is exactly a box of `x#a` and `x#b`.
    auto split = [&](const Lin& l, u64 j) {
        auto nz = std::ranges::count_if(l.c, [](int64_t ci) { return ci != 0; });
        if (nz != 2 || l.sym || l.k != 0 || std::ranges::count(l.c, 1) != 1) return false;
        auto b = u64(std::ranges::find(l.c, 1) - l.c.begin());
        auto a = u64(std::ranges::find_if(l.c, [](int64_t ci) { return ci > 1; }) - l.c.begin());
        if (a == *ln || seen[a] || seen[b] || quot[a].b || rems[a].b || quot[b].b || rems[b].b) return false;
        auto q = u64(l.c[a]);
        if (Lit::isa(S->proj(*ln, b)) != q) return false;
        auto y_lo = lo->proj(*lm, j), y_ext = ext->proj(*lm, j);
        auto bq  = w.lit_nat(q);
        auto div = [&](const Def* x) { return w.call(core::nat::div, Defs{x, bq}); };
        if (Lit::isa(y_ext) == 1) {
            los[a] = div(y_lo), exts[a] = caps[a] = w.lit_nat_1();
            los[b] = w.call(core::nat::rem, Defs{y_lo, bq}), exts[b] = caps[b] = w.lit_nat_1();
        } else if (auto e = Lit::isa(y_ext);
                   e && *e % q == 0 && w.call(core::nat::rem, Defs{y_lo, bq}) == w.lit_nat_0()) {
            los[a]  = div(y_lo);
            caps[a] = region.min(w.lit_nat(*e / q), S->proj(*ln, a));
            exts[a] = region.sub(region.min(region.add(los[a], caps[a]), S->proj(*ln, a)), los[a]);
            los[b] = w.lit_nat_0(), exts[b] = caps[b] = bq;
        } else {
            return false;
        }
        seen[a] = seen[b] = true;
        return true;
    };
    for (u64 j = 0; j != *lm; ++j) {
        if (auto o = Axm::isa<affine::semiop>(body->proj(*lm, j));
            o && (o.id() == semiop::floordiv || o.id() == semiop::rem)) {
            auto [a, c] = o->args<2>();
            auto x      = region.eval(a);
            auto b      = Lit::isa<u64>(c);
            if (!x || !x->lin || !one(*x->lin) || x->lin->k != 0 || x->lin->sym || !b || *b == 0) {
                exact = false;
                continue;
            }
            auto i  = u64(std::ranges::find(x->lin->c, 1) - x->lin->c.begin());
            auto& d = o.id() == semiop::floordiv ? quot[i] : rems[i];
            if (seen[i] || d.b) {
                exact = false;
                continue;
            }
            d = Split{*b, lo->proj(*lm, j), ext->proj(*lm, j)};
            continue;
        }
        auto itv = region.eval(body->proj(*lm, j));
        if (itv && itv->lin && split(*itv->lin, j)) continue;
        if (!itv || !itv->lin || !one(*itv->lin)) {
            exact = false;
            continue;
        }
        auto& l = *itv->lin;
        auto i  = u64(std::ranges::find(l.c, 1) - l.c.begin());
        if (seen[i]) {
            exact = false;
            continue;
        }
        seen[i] = true;
        // x#i = y#j − (sym + k); shifting both ends keeps the box exact where the lower one saturates at 0.
        auto shift = [&](const Def* y) {
            if (l.sym) y = region.sub(y, l.sym);
            return l.k >= 0 ? region.sub(y, w.lit_nat(u64(l.k))) : region.add(y, w.lit_nat(u64(-l.k)));
        };
        auto y_lo = lo->proj(*lm, j);
        auto hi   = region.min(shift(region.add(y_lo, ext->proj(*lm, j))), S->proj(*ln, i));
        los[i]    = shift(y_lo);
        exts[i]   = region.sub(hi, los[i]);
        caps[i]   = region.min(ext->proj(*lm, j), S->proj(*ln, i));
    }

    for (u64 i = 0; i != *ln; ++i) {
        auto &q = quot[i], &rm = rems[i];
        if (!q.b && !rm.b) continue;
        if (seen[i] || !q.b || (rm.b && rm.b != q.b)) {
            exact = false;
            continue;
        }
        auto b     = w.lit_nat(q.b);
        auto whole = !rm.b || (Lit::isa(rm.lo) == 0 && Lit::isa(rm.ext) >= q.b);
        auto start = region.mul(q.lo, b);
        auto size  = region.mul(q.ext, b);
        if (!whole) {
            if (Lit::isa(q.ext) != 1) {
                exact = false;
                continue;
            }
            start = region.add(start, rm.lo);
            size  = rm.ext;
        }
        los[i]  = start;
        exts[i] = region.sub(region.min(region.add(start, size), S->proj(*ln, i)), start);
        caps[i] = region.min(size, S->proj(*ln, i));
    }

    return w.tuple({w.tuple(los), w.tuple(exts), w.tuple(caps), exact ? w.lit_tt() : w.lit_ff()});
}

MIM_affine_NORMALIZER_IMPL

} // namespace mim::plug::affine
