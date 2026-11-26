#include <algorithm>
#include <array>

#include <fe/format.h>

#include <mim/check.h>
#include <mim/def.h>
#include <mim/driver.h>
#include <mim/plugin.h>
#include <mim/tuple.h>
#include <mim/world.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/btensor/btensor.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

namespace {

const Def* read_through(const Def* base, const Def* index) {
    auto& w = base->world();
    const Def *input, *s_in, *s_out, *rank;
    bool wraps;
    if (auto rep = Axm::isa<tensor::repeat>(base)) {
        auto [Tr, in, out] = rep->callee()->as<App>()->uncurry_args<3>();
        input = rep->arg(), s_in = in, s_out = out, rank = Tr->proj(2, 1), wraps = true;
    } else if (auto bc = Axm::isa<tensor::broadcast>(base)) {
        auto [in, out, i] = bc->args<3>();
        auto [_, r]       = bc->callee()->as<App>()->args<2>();
        input = i, s_in = in, s_out = out, rank = r, wraps = false;
    } else {
        return nullptr;
    }
    auto r = Lit::isa<u64>(rank);
    if (!r) return nullptr;

    auto mapped = DefVec(*r);
    for (u64 d = 0; d != *r; ++d) {
        auto in  = s_in->proj(*r, d);
        auto out = s_out->proj(*r, d);
        auto idx = index->proj(*r, d);
        if (in == out)
            mapped[d] = idx;
        else if (Lit::isa(in) == 1)
            mapped[d] = w.lit_idx(1, 0);
        else if (auto e = Lit::isa(in), x = Lit::isa(idx); wraps && e && *e && x)
            mapped[d] = w.lit_idx(*e, *x % *e);
        else
            return nullptr;
    }
    return op_get(input, w.tuple(mapped));
}

} // namespace

const Def* normalize_get(const Def*, const Def* callee, const Def* arg) {
    auto [tensor, index] = arg->projs<2>();
    if (auto array = Axm::isa<tensor::from_array>(tensor)) return arg->world().extract(array->arg(), index);
    if (auto scalar = Axm::isa<tensor::scalar>(tensor)) return scalar->arg();
    if (auto set = Axm::isa<tensor::set>(tensor)) {
        auto [base, written, value] = set->args<3>();
        if (written == index) return value;
        if (auto rank = Lit::isa<u64>(callee->as<App>()->arg(3, 0))) {
            auto a = lit_projs(written, *rank);
            auto b = lit_projs(index, *rank);
            if (a && b && *a != *b) return op_get(base, index);
        }
    }
    return read_through(tensor, index);
}

const Def* normalize_set(const Def*, const Def* callee, const Def* arg) {
    auto [tensor, index, value] = arg->projs<3>();
    if (auto array = Axm::isa<tensor::from_array>(tensor)) {
        auto data = array->arg();
        auto rank = Lit::isa<u64>(callee->as<App>()->arg(3, 0));
        if (data->is_closed() && rank && lit_projs(index, *rank) && value->isa<Lit>()) {
            auto& w = arg->world();
            auto op = w.app(w.annex<tensor::from_array>(), callee->as<App>()->arg());
            return w.app(op, w.insert(data, index, value));
        }
    }
    return nullptr;
}

const Def* normalize_scalar(const Def* type, const Def*, const Def* arg) {
    if (auto item = Axm::isa<tensor::item>(arg))
        if (item->arg()->type() == type) return item->arg();
    return nullptr;
}

const Def* normalize_item(const Def*, const Def*, const Def* arg) {
    if (auto scalar = Axm::isa<tensor::scalar>(arg)) return scalar->arg();
    if (auto array = Axm::isa<tensor::from_array>(arg)) return array->arg();
    if (auto set = Axm::isa<tensor::set>(arg)) return set->arg()->proj(3, 2);
    return nullptr;
}

const Def* normalize_broadcast(const Def* type, const Def* c, const Def* arg) {
    auto& w = c->world();

    auto [s_in, s_out, input] = arg->projs<3>();
    auto callee               = c->as<App>();
    auto [T, r]               = callee->args<2>();
    w.log().d("broadcast: input = {}: {}, T = {}, r = {}, s_in = {}, s_out = {}", input, input->type(), T, r, s_in,
              s_out);

    if (s_in == s_out || input->type() == type) return input;

    return nullptr;
}

const Def* normalize_broadcast_in_dim(const Def*, const Def*, const Def*) { return nullptr; }

const Def* normalize_repeat(const Def* type, const Def* c, const Def* arg) {
    // Identity repeat: if the input and output shapes agree, the repeat is a no-op.
    auto [Tr, s_in, s_out] = c->as<App>()->uncurry_args<3>();
    if (s_in == s_out || arg->type() == type) return arg;
    return nullptr;
}

const Def* normalize_reshape(const Def* type, const Def* c, const Def* arg) {
    // Identity reshape: if the input and output shapes agree, the reshape is a no-op.
    auto [Trr, s_in, s_out] = c->as<App>()->uncurry_args<3>();
    if (s_in == s_out || arg->type() == type) return arg;
    return nullptr;
}

const Def* normalize_transpose(const Def*, const Def* c, const Def* arg) {
    auto callee = c->as<App>()->decurry();
    auto perm   = lit_perm(callee->arg());
    if (!perm) return nullptr;
    auto r = perm->size();
    if (std::ranges::equal(*perm, std::views::iota(u64(0), u64(r)))) return arg;

    auto app = Axm::isa<tensor::transpose>(arg);
    if (!app) return nullptr;
    auto inner = transpose_perm(app);
    if (!inner || inner->size() != r) return nullptr;

    // Axis `j` of the inner input ends up at `perm#(inner#j)`; an identity composite collapses in the rebuilt app.
    auto& w   = c->world();
    auto comp = DefVec(r, [&](size_t j) { return w.lit_idx(r, (*perm)[(*inner)[j]]); });
    return w.app(w.app(w.app(callee->callee(), w.tuple(comp)), app->decurry()->arg()), app->arg());
}

const Def* normalize_slice(const Def*, const Def* c, const Def* arg) {
    // Identity slice: every axis starts at 0 with step 1 and keeps its full extent (s_out == s_in) -> the input itself.
    auto [Tr, s_in, params]   = c->as<App>()->uncurry_args<3>();
    auto [start, step, s_out] = params->projs<3>();
    if (s_out != s_in) return nullptr;
    auto r = Lit::isa<u64>(Tr->proj(2, 1));
    if (!r) return nullptr;
    for (u64 d = 0; d != *r; ++d) {
        auto st = Lit::isa<u64>(start->proj(*r, d));
        auto sp = Lit::isa<u64>(step->proj(*r, d));
        if (!st || *st != 0 || !sp || *sp != 1) return nullptr;
    }
    return arg;
}

const Def* normalize_flip(const Def*, const Def*, const Def*) { return nullptr; }

const Def* normalize_pad(const Def*, const Def* c, const Def* arg) {
    // Identity pad: every axis has lo == hi == 0 (so s_out == s_in) -> the input itself (the fill value is irrelevant).
    auto [Tr, s_in, params] = c->as<App>()->uncurry_args<3>();
    auto [mode, lo, hi]     = params->projs<3>();
    auto r                  = Lit::isa<u64>(Tr->proj(2, 1));
    if (!r) return nullptr;
    for (u64 d = 0; d != *r; ++d) {
        auto l = Lit::isa<u64>(lo->proj(*r, d));
        auto h = Lit::isa<u64>(hi->proj(*r, d));
        if (!l || *l != 0 || !h || *h != 0) return nullptr;
    }
    return arg->proj(2, 0); // input (arg = (input, value))
}

const Def* normalize_concat(const Def*, const Def*, const Def*) { return nullptr; }

const Def* normalize_if_static(const Def*, const Def*, const Def* arg) {
    // `tensor.if_static (k, s, d)` picks `s` once `k` has folded to a literal; a still-symbolic `k`
    // keeps the App stuck, and the tensor lowerings residualize it to `d` (by lowering time,
    // undecided means runtime).
    auto [k, s, d] = arg->projs<3>();
    if (Lit::isa(k)) return s;
    return nullptr;
}

const Def* normalize_fastest_axis(const Def*, const Def*, const Def* arg) {
    // `tensor.fastest_axis (r, t)` reflects which axis of `t` is the fastest-varying (unit-stride)
    // axis of the tensor actually read once `fuse_tensor`'s read-through has absorbed a pure
    // re-indexed read behind `t`: without one, `t`'s own last axis; behind one, found by evaluating
    // the read's access map on distinct `affine.lit` markers — normalization folds the map's
    // extracts over the marker tuple, and a last component that does not fold back to a marker
    // (reshape arithmetic) stays unknown. Unknown answers the sentinel `r`.
    auto& w     = arg->world();
    auto [r, t] = arg->projs<2>();
    auto r_l    = Lit::isa<u64>(r);
    if (!r_l || *r_l == 0) return r;
    auto pr = is_pure_read(t);
    if (!pr) return w.lit_nat(*r_l - 1);
    // One level only: a source that is itself absorbed would need the composed analysis.
    if (is_pure_read(pr->src) || Axm::isa<tensor::broadcast>(pr->src)) return r;
    auto r_src = Lit::isa<u64>(pr->map->type()->as<Pi>()->codom()->arity());
    if (!r_src || *r_src == 0) return r;
    // `r` is not type-coupled to `t`; a mismatched rank must answer unknown, not break the app below.
    auto r_dom = Lit::isa<u64>(pr->map->type()->as<Pi>()->dom()->arity());
    if (!r_dom || *r_dom != *r_l) return r;
    // Markers start at 1: 0 is what a broadcast map's `o#d · 0` folds to, so it must not be one.
    auto markers = DefVec(*r_l, [&](size_t i) { return w.call<affine::lit>(w.lit_nat(i + 1)); });
    auto last    = w.app(pr->map, w.tuple(markers))->proj(*r_src, *r_src - 1);
    auto c       = Axm::isa<affine::lit>(last);
    if (!c) return r;
    auto v = Lit::isa<u64>(c->arg());
    if (!v || *v < 1 || *v > *r_l) return r;
    return w.lit_nat(*v - 1);
}

/// The vector widths `-X tensor:vec-width` accepts.
constexpr std::array Vec_widths = {1_u64, 4_u64, 8_u64, 16_u64};

const Def* normalize_vec_width(const Def*, const Def*, const Def* arg) {
    // Read off the Driver so one graph can be re-emitted for another target by another Driver.
    auto& w = arg->world();
    auto n  = w.driver().arg_nat("tensor", "vec-width");
    if (!n) return arg;
    if (std::ranges::find(Vec_widths, *n) == Vec_widths.end()) {
        w.log().w("ignoring `-X tensor:vec-width={}`: not one of {}", *n, fe::Join(Vec_widths, ", "));
        return arg;
    }
    return w.lit_nat(*n);
}

namespace {

/// Translates an access map's `affine.*` terms over the loop variables `xs` into `tensor.AffE` trees.
/// `scan` checks a term and collects its run-time `affine.lit`s as symbols; `build` then emits the tree.
class AffReflect {
public:
    AffReflect(World& w, const Def* n, Defs xs)
        : w_(w)
        , n_(n)
        , xs_(xs) {}

    bool scan(const Def* t) {
        if (auto lit = Axm::isa<affine::lit>(t)) {
            if (!Lit::isa(lit->arg()) && std::ranges::find(syms_, lit->arg()) == syms_.end())
                syms_.emplace_back(lit->arg());
            return true;
        }
        if (dim(t)) return true;
        if (auto add = Axm::isa(affine::op::add, t)) {
            auto [l, r] = add->args<2>();
            return scan(l) && scan(r);
        }
        if (auto op = Axm::isa<affine::semiop>(t)) {
            auto [l, c] = op->args<2>();
            auto k      = Lit::isa(c);
            return k && *k != 0 && scan(l);
        }
        if (auto lin = Axm::isa<affine::linearize>(t)) {
            auto [idxs, s] = lin->args<2>();
            auto r         = Lit::isa(s->arity());
            if (!r || !lit_projs(s, *r)) return false;
            for (size_t k = 0; k != *r; ++k)
                if (!scan(idxs->proj(*r, k))) return false;
            return true;
        }
        if (auto [l, s, k] = delinearize(t); l) return lit_projs(s, *Lit::isa(s->arity())) && scan(l);
        return false;
    }

    const Def* build(const Def* t) {
        if (auto lit = Axm::isa<affine::lit>(t)) {
            if (Lit::isa(lit->arg())) return ctor<tensor::aff_lit>(lit->arg());
            auto k = std::ranges::find(syms_, lit->arg()) - syms_.begin();
            return ctor<tensor::aff_sym>(w_.lit_idx(syms_.size(), k));
        }
        if (auto j = dim(t)) return ctor<tensor::aff_dim>(w_.lit_idx(xs_.size(), *j));
        if (auto add = Axm::isa(affine::op::add, t)) {
            auto [l, r] = add->args<2>();
            return ctor<tensor::aff_add>({build(l), build(r)});
        }
        if (auto op = Axm::isa<affine::semiop>(t)) {
            auto [l, c] = op->args<2>();
            auto x      = build(l);
            switch (op.id()) {
                case affine::semiop::mul: return ctor<tensor::aff_mul>({x, c});
                case affine::semiop::floordiv: return ctor<tensor::aff_fdiv>({x, c});
                case affine::semiop::ceildiv: return ctor<tensor::aff_cdiv>({x, c});
                case affine::semiop::rem: return ctor<tensor::aff_mod>({x, c});
            }
        }
        if (auto lin = Axm::isa<affine::linearize>(t)) {
            auto [idxs, s] = lin->args<2>();
            auto r         = *Lit::isa(s->arity());
            const Def* sum = nullptr;
            u64 stride     = 1;
            for (size_t k = r; k-- != 0;) {
                auto x = build(idxs->proj(r, k));
                if (stride != 1) x = ctor<tensor::aff_mul>({x, w_.lit_nat(stride)});
                sum = sum ? ctor<tensor::aff_add>({x, sum}) : x;
                stride *= *Lit::isa(s->proj(r, k));
            }
            return sum;
        }
        auto [l, s, k] = delinearize(t);
        auto m         = *Lit::isa(s->arity());
        u64 stride     = 1;
        for (size_t j = k + 1; j != m; ++j)
            stride *= *Lit::isa(s->proj(m, j));
        auto x = build(l);
        if (stride != 1) x = ctor<tensor::aff_fdiv>({x, w_.lit_nat(stride)});
        return ctor<tensor::aff_mod>({x, s->proj(m, k)});
    }

    /// The `tensor.Aff` with the rows `rows`, built after scanning them.
    const Def* map(Defs rows) {
        auto s   = w_.lit_nat(syms_.size());
        auto map = w_.app(w_.annex<tensor::aff_map>(), {n_, w_.lit_nat(rows.size()), s});
        return w_.app(map, {w_.tuple(syms_), w_.tuple(DefVec(rows.size(), [&](size_t i) { return build(rows[i]); }))});
    }

private:
    std::optional<size_t> dim(const Def* t) const {
        if (auto it = std::ranges::find(xs_, t); it != xs_.end()) return it - xs_.begin();
        return {};
    }

    /// `delinearize (l, s) # k` as `(l, s, k)`, or `l == nullptr`.
    static std::tuple<const Def*, const Def*, size_t> delinearize(const Def* t) {
        if (auto ex = t->isa<Extract>())
            if (auto del = Axm::isa<affine::delinearize>(ex->tuple()))
                if (auto k = Lit::isa(ex->index()); k && Lit::isa(del->arg(2, 1)->arity())) {
                    auto [l, s] = del->args<2>();
                    return {l, s, *k};
                }
        return {nullptr, nullptr, 0};
    }

    template<class Id>
    const Def* ctor(Defs args) {
        return w_.app(w_.app(w_.annex<Id>(), {n_, w_.lit_nat(syms_.size())}), args);
    }
    template<class Id>
    const Def* ctor(const Def* arg) {
        return w_.app(w_.app(w_.annex<Id>(), {n_, w_.lit_nat(syms_.size())}), arg);
    }

    World& w_;
    const Def* n_;
    Defs xs_;
    DefVec syms_;
};

} // namespace

const Def* pe_cpu_plan(const App* callee, PeCpu* plan) {
    auto& w = callee->world();
    // `run_levels` tiles for a CPU; a GPU pipeline launches the contraction as a kernel instead.
    if (!w.driver().arg_nat("tensor", "pe-cpu").value_or(1) || w.driver().is_loaded("gpu")) return w.lit_ff();
    auto mr    = btensor::MapRedPost::of(callee);
    auto ranks = mr.ranks();
    if (!ranks) return nullptr;
    if (ranks->ro < 2 || ranks->rr != 1 || ranks->nis != 2) return w.lit_ff();
    u64 ro = ranks->ro;
    u64 rn = ranks->nloops();
    // Extents may be run-time values; `run_levels` splits those with `Loop` or `Peel` and computes edge tiles in scratch.
    if (!lit_projs(mr.Ris, 2)) return nullptr;
    for (u64 d = 0; d != rn; ++d)
        if (Lit::isa(mr.Sr->proj(rn, d)) == 0) return w.lit_ff();
    if (mr.To == w.type_nat()) return w.lit_ff();
    auto bytes = Lit::isa(w.app(w.annex(core::trait::size), mr.To));
    if (!bytes) return mr.To->is_closed() ? w.lit_ff() : nullptr;
    if (*bytes == 0 || !Check::test_alpha(mr.init->type(), mr.To)) return w.lit_ff();
    if (!Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>())) return w.lit_ff();
    // An explicit schedule such as `dot_schedule` stays.
    if (!Check::test_alpha(mr.sched, w.app(w.annex<btensor::mk_sched>(), {mr.Rn, w.lit_nat_0()}))) return w.lit_ff();

    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto probe  = w.mut_lam(domain, domain);
    auto vars   = probe->var()->projs(rn);
    // The output is a permutation of the parallel loops; only the identity also handles run-time extents, so a
    // permutation waits for literal ones.
    auto out  = w.app(mr.map_out, probe->var())->projs(ro);
    bool perm = false;
    for (u64 k = 0; k != ro; ++k) {
        auto it = std::ranges::find(vars, out[k]);
        if (it == vars.end() || u64(it - vars.begin()) >= ro || std::ranges::count(out, out[k]) != 1) return w.lit_ff();
        auto d = u64(it - vars.begin());
        if (mr.So->proj(ro, k) != mr.Sr->proj(rn, d)) return w.lit_ff();
        perm |= d != k;
    }
    if (perm)
        for (u64 d = 0; d != rn; ++d)
            if (!Lit::isa(mr.Sr->proj(rn, d))) return nullptr;

    auto maps = mr.maps->projs<2>();
    auto dep  = w.tuple(DefVec(2, [&](size_t i) {
        auto base = w.app(maps[i], probe->var());
        return w.tuple(DefVec(rn, [&](size_t d) {
            auto sub = vars;
            sub[d]   = w.call<affine::lit>(w.lit_nat_0());
            return w.lit_bool(!Check::test_alpha(base, w.app(maps[i], w.tuple(sub))));
        }));
    }));
    auto a    = w.lit_idx(2, dep->proj(2, 0)->proj(rn, ro - 2) == w.lit_tt() ? 0 : 1);
    if (w.app(w.app(w.annex<tensor::cpu_covers>(), {mr.Rn, mr.Ro}), {a, dep}) != w.lit_tt()) return w.lit_ff();

    auto affs = DefVec(2);
    for (size_t i = 0; i != 2; ++i) {
        auto r    = *Lit::isa(mr.Ris->proj(2, i));
        auto rows = w.app(maps[i], probe->var())->projs(r);
        auto refl = AffReflect(w, mr.Rn, vars);
        // A map with symbolic parameters (a generic convolution's) may still become affine once they are literals.
        for (auto row : rows)
            if (!refl.scan(row)) return maps[i]->is_closed() ? w.lit_ff() : nullptr;
        affs[i] = refl.map(rows);
    }
    auto b = 1 - Lit::as(a);
    auto unit_b
        = w.app(w.app(w.annex<tensor::aff_unit>(), {mr.Rn, mr.Ris->proj(2, b)}), {affs[b], w.lit_idx(rn, ro - 1)});
    if (plan) *plan = {*bytes, a, dep, unit_b, w.tuple(affs)};
    return w.lit_tt();
}

namespace {

// Whether `def` reaches `target` without passing `stop` or entering a mutable.
bool reaches(const Def* def, const Def* target, const Def* stop) {
    auto seen = DefSet();
    auto todo = DefVec{def};
    while (!todo.empty()) {
        auto d = todo.back();
        todo.pop_back();
        if (d == target) return true;
        if (d == stop || d->isa_mut() || !seen.emplace(d).second) continue;
        for (auto op : d->deps())
            todo.emplace_back(op);
    }
    return false;
}

// The operation `⊕` of a combiner `λ (acc, xs). acc ⊕ e` that reassociates exactly or by its math mode and has `init`
// as its unit, or `nullptr`.
const Def* reduce_monoid(const Def* comb, const Def* init) {
    auto& w   = comb->world();
    auto lam  = comb->isa_mut<Lam>();
    auto body = Lam::isa_ret_arg(comb);
    auto app  = body ? body->isa<App>() : nullptr;
    if (!app || app->arg()->num_projs() != 2) return nullptr;
    auto args = lam->var(2, 0);
    auto acc = args->proj(2, 0), ins = args->proj(2, 1);
    auto [a, b] = app->arg()->projs<2>();
    auto e      = a == acc ? b : b == acc ? a : nullptr;
    if (!e || reaches(e, acc, ins) || reaches(e, args, ins)) return nullptr;
    auto unit = [&](f64 v) {
        auto width = math::isa_f(init->type());
        return width && init == math::lit_f(w, *width, v);
    };
    if (auto ar = Axm::isa<math::arith>(body)) {
        auto mode = Lit::isa(ar->callee()->as<App>()->callee()->as<App>()->arg());
        if (!mode || !(*mode & std::to_underlying(math::Mode::reassoc))) return nullptr;
        if ((ar.id() == math::arith::add && unit(0.0)) || (ar.id() == math::arith::mul && unit(1.0)))
            return app->callee();
    } else if (auto ex = Axm::isa<math::extrema>(body)) {
        bool max = ex.id() == math::extrema::iM || ex.id() == math::extrema::IM;
        if (unit(max ? -std::numeric_limits<f64>::infinity() : std::numeric_limits<f64>::infinity()))
            return app->callee();
    } else if (auto wr = Axm::isa<core::wrap>(body)) {
        auto mode = Lit::isa(wr->callee()->as<App>()->callee()->as<App>()->arg());
        if (mode != 0) return nullptr;
        if ((wr.id() == core::wrap::add && Lit::isa(init) == 0) || (wr.id() == core::wrap::mul && Lit::isa(init) == 1))
            return app->callee();
    }
    return nullptr;
}

} // namespace

bool pe_generic_plan(const App* callee, PeGeneric* plan) {
    auto& w    = callee->world();
    auto mr    = btensor::MapRedPost::of(callee);
    auto ranks = mr.ranks();
    if (!ranks || ranks->ro < 1 || ranks->nis < 1) return false;
    u64 ro = ranks->ro, rn = ranks->nloops(), nis = ranks->nis;
    if (!lit_projs(mr.Ris, nis)) return false;
    for (u64 d = ro; d != rn; ++d)
        if (!Lit::isa(mr.Sr->proj(rn, d))) return false;
    if (mr.To == w.type_nat() || !Check::test_alpha(mr.init->type(), mr.To)) return false;

    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto probe  = w.mut_lam(domain, domain);
    auto vars   = probe->var()->projs(rn);
    // The output is a permutation of the parallel loops, which only literal extents can tile.
    auto out = w.app(mr.map_out, probe->var())->projs(ro);
    u64 wdim = ro - 1;
    for (u64 k = 0; k != ro; ++k) {
        auto it = std::ranges::find(vars, out[k]);
        if (it == vars.end() || u64(it - vars.begin()) >= ro || std::ranges::count(out, out[k]) != 1) return false;
        auto d = u64(it - vars.begin());
        if (mr.So->proj(ro, k) != mr.Sr->proj(rn, d)) return false;
        if (d != k)
            for (u64 e = 0; e != rn; ++e)
                if (!Lit::isa(mr.Sr->proj(rn, e))) return false;
        if (k == ro - 1) wdim = d;
    }

    auto maps = mr.maps->projs(nis);
    auto affs = DefVec(nis);
    // Unit stride of every input along a loop: along the innermost one the plain nest is already contiguous; along the
    // last output loop with a strided reduction inside, a row of accumulators makes it so.
    auto unit = [&](size_t i, u64 d) {
        auto aff_unit = w.app(w.annex<tensor::aff_unit>(), {mr.Rn, mr.Ris->proj(nis, i)});
        return w.app(aff_unit, {affs[i], w.lit_idx(rn, d)}) == w.lit_tt();
    };
    bool red_unit = true, out_unit = true;
    u64 rdim = wdim;
    for (size_t i = 0; i != nis; ++i) {
        auto r    = *Lit::isa(mr.Ris->proj(nis, i));
        auto rows = w.app(maps[i], probe->var())->projs(r);
        auto refl = AffReflect(w, mr.Rn, vars);
        for (auto row : rows)
            if (!refl.scan(row)) return false;
        affs[i] = refl.map(rows);
        red_unit &= unit(i, rn - 1);
        out_unit &= unit(i, ro - 1);
        auto sub   = vars;
        sub[wdim]  = w.call<affine::lit>(w.lit_nat_0());
        bool reads = !Check::test_alpha(w.app(maps[i], probe->var()), w.app(maps[i], w.tuple(sub)));
        if (rdim == wdim && reads && !unit(i, wdim))
            for (u64 d = 0; d != ro; ++d)
                if (unit(i, d)) rdim = d;
    }
    // An outer product: one input reads the second last output loop but not the last, the other the last at unit stride
    // but not the second last.
    bool outer = false;
    if (nis == 2 && ro >= 2) {
        auto reads = [&](size_t i, u64 d) {
            auto sub = vars;
            sub[d]   = w.call<affine::lit>(w.lit_nat_0());
            return !Check::test_alpha(w.app(maps[i], probe->var()), w.app(maps[i], w.tuple(sub)));
        };
        for (size_t a = 0; a != 2; ++a) {
            auto b = 1 - a;
            outer |= reads(a, ro - 2) && !reads(a, ro - 1) && unit(b, ro - 1) && !reads(b, ro - 2);
        }
    }
    auto merge = reduce_monoid(mr.comb, mr.init);
    auto bytes = Lit::isa(w.app(w.annex(core::trait::size), mr.To));
    u64 lanes  = merge && bytes && *bytes != 0 ? btensor::vec_regs(w.driver()).bytes / *bytes : 0;
    if (plan)
        *plan = {w.tuple(affs), w.lit_bool(red_unit), w.lit_bool(out_unit),         w.lit_nat(wdim),  w.lit_nat(rdim),
                 merge,         w.lit_nat(lanes),     w.lit_nat(bytes.value_or(0)), w.lit_bool(outer)};
    return true;
}

const Def* normalize_pe_cpu_candidate(const Def*, const Def*, const Def* producer) {
    if (auto app = Axm::isa<tensor::map_reduce_post>(producer)) return pe_cpu_plan(app->callee()->as<App>());
    return nullptr;
}

const Def* normalize_cpu_knobs(const Def*, const Def* c, const Def*) {
    auto& w   = c->world();
    auto knob = [&](const char* name, u64 fallback) {
        return w.lit_nat(w.driver().arg_nat("tensor", name).value_or(fallback));
    };
    auto vr = btensor::vec_regs(w.driver());
    return w.tuple({w.lit_nat(vr.bytes), knob("group-registers", vr.count), knob("group-cache", 262144),
                    knob("cpu-mr", 0), knob("cpu-nr", 0), knob("cpu-mc", 0), knob("cpu-nc", 0), knob("cpu-kc", 0),
                    knob("cpu-order", 0), knob("cpu-dyn-k", 1), knob("cpu-pack-b", 0), knob("cpu-conv", 0)});
}

MIM_tensor_NORMALIZER_IMPL

} // namespace mim::plug::tensor
