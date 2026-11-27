
#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <vector>

#include <fe/format.h>

#include <mim/check.h>
#include <mim/def.h>
#include <mim/driver.h>
#include <mim/nom.h>
#include <mim/plugin.h>
#include <mim/tuple.h>
#include <mim/variant.h>
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
    // `tensor.fastest_axis (r, t)` names the fastest-varying (unit-stride) axis of the tensor actually read once `fuse_tensor`'s read-through has absorbed a pure re-indexed read behind `t`:
    // without one, `t`'s own last axis; behind one, the loop the last row of the read's access map is, if it is one.
    // Unknown answers the sentinel `r`.
    auto& w     = arg->world();
    auto [r, t] = arg->projs<2>();
    auto r_l    = Lit::isa<u64>(r);
    if (!r_l || *r_l == 0) return r;
    auto pr = is_pure_read(t);
    if (!pr) return w.lit_nat(*r_l - 1);
    // One level only: a source that is itself absorbed would need the composed analysis.
    if (is_pure_read(pr->src) || Axm::isa<tensor::broadcast>(pr->src)) return r;
    // `r` is not type-coupled to `t`; a mismatched rank must answer unknown.
    if (Lit::isa<u64>(pr->n) != r_l) return r;
    auto d = aff_last_dim(pr->aff, pr->n, pr->R);
    return d ? w.lit_nat(*d) : r;
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

/// Rebuilds an `Aff` with its literal values in its rows and its run-time values as symbols, in the order they first occur.
class AffCanon {
public:
    AffCanon(World& w, const Def* n, const Def* vals, size_t s)
        : w_(w)
        , n_(n)
        , vals_(vals)
        , s_(s) {}

    bool scan(const Def* e) {
        auto inj = decode(e);
        if (!inj) return false;
        auto v = inj->value();
        switch (inj->index()) {
            case 0: return true;
            case 1:
            case 2: return sym(value(inj)), true;
            case 3:
            case 8: return scan(v->proj(2, 0)) && scan(v->proj(2, 1));
            default: {
                auto k = decode(v->proj(2, 1));
                if (!k) return false;
                sym(value(k));
                return scan(v->proj(2, 0));
            }
        }
    }

    const Def* build(const Def* e) {
        auto inj = decode(e);
        auto v   = inj->value();
        switch (inj->index()) {
            case 0: return ctor<tensor::aff_dim>(v);
            case 1:
            case 2: {
                auto c = value(inj);
                return Lit::isa(c) ? ctor<tensor::aff_lit>(c) : ctor<tensor::aff_sym>(idx(index(c)));
            }
            case 3: return ctor<tensor::aff_add>({build(v->proj(2, 0)), build(v->proj(2, 1))});
            case 8: return ctor<tensor::aff_sub>({build(v->proj(2, 0)), build(v->proj(2, 1))});
            default: {
                auto c = value(decode(v->proj(2, 1)));
                auto l = Lit::isa(c);
                // `x · 0` reads index 0 along a broadcast axis; a factor or divisor 1 is no operation.
                if (inj->index() == 4 && l == 0) return ctor<tensor::aff_lit>(w_.lit_nat_0());
                auto x = build(v->proj(2, 0));
                if (inj->index() < 7 && l == 1) return x;
                auto s    = w_.lit_nat(syms_.size());
                auto coef = l ? w_.app(w_.app(w_.annex<tensor::aff_num>(), s), c)
                              : w_.app(w_.app(w_.annex<tensor::aff_var>(), s), idx(index(c)));
                switch (inj->index()) {
                    case 4: return ctor<tensor::aff_mul>({x, coef});
                    case 5: return ctor<tensor::aff_fdiv>({x, coef});
                    case 6: return ctor<tensor::aff_cdiv>({x, coef});
                    default: return ctor<tensor::aff_mod>({x, coef});
                }
            }
        }
    }

    const DefVec& syms() const { return syms_; }

private:
    static const Inj* decode(const Def* e) {
        auto name = e->isa<Name>();
        return name ? name->op()->isa<Inj>() : nullptr;
    }
    /// The value of a `Num`/`Lit` (index 0 resp. 2) or the value a `Var`/`Sym` refers to.
    const Def* value(const Inj* k) const {
        auto v = k->value();
        if (k->index() == 0 || k->index() == 2) return v;
        return vals_->proj(s_, *Lit::isa(v));
    }
    void sym(const Def* v) {
        if (!Lit::isa(v) && std::ranges::find(syms_, v) == syms_.end()) syms_.emplace_back(v);
    }
    size_t index(const Def* v) const { return std::ranges::find(syms_, v) - syms_.begin(); }
    const Def* idx(size_t k) const { return w_.lit_idx(syms_.size(), k); }

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
    const Def* vals_;
    size_t s_;
    DefVec syms_;
};

} // namespace

const Def* pe_cpu_plan(const App* callee, PeCpu* plan) {
    auto& w = callee->world();
    // `run_levels` tiles for a CPU; a GPU pipeline launches the contraction as a kernel instead.
    if (!w.driver().arg_nat("tensor", "pe-cpu").value_or(1) || w.driver().is_loaded("gpu")) return w.lit_ff();
    auto mr    = map_red(callee);
    auto ranks = mr.ranks();
    if (!ranks) return nullptr;
    if (ranks->ro < 2 || ranks->rr != 1 || ranks->nis != 2) return w.lit_ff();
    u64 ro = ranks->ro;
    u64 rn = ranks->nloops();
    // Extents may be run-time values; `run_levels` splits those with `Loop` or `Peel` and computes edge tiles in
    // scratch.
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

    // The output is a permutation of the parallel loops; only the identity also handles run-time extents, so a
    // permutation waits for literal ones.
    auto raw = btensor::MapRedPost::of(callee);
    // Maps a generic impl builds from its parameters decide once they are literals.
    for (auto aff : {raw.map_out, raw.maps->proj(2, 0), raw.maps->proj(2, 1)})
        if (aff->isa<Name>() && !aff_static(aff)) return nullptr;
    auto loops = out_loops(raw.map_out, mr.Rn, mr.Ro);
    if (!loops) return w.lit_ff();
    bool perm = false;
    for (u64 k = 0; k != ro; ++k) {
        if (mr.So->proj(ro, k) != mr.Sr->proj(rn, (*loops)[k])) return w.lit_ff();
        perm |= (*loops)[k] != k;
    }
    if (perm)
        for (u64 d = 0; d != rn; ++d)
            if (!Lit::isa(mr.Sr->proj(rn, d))) return nullptr;

    auto affs = raw.maps->projs<2>();
    for (size_t i = 0; i != 2; ++i) {
        auto aff = aff_canon(affs[i], mr.Rn, mr.Ris->proj(2, i));
        if (!aff) return affs[i]->is_closed() ? w.lit_ff() : nullptr;
        affs[i] = aff;
    }
    auto facts = w.app(w.annex<tensor::cpu_facts>(), {mr.Rn, mr.Ro, mr.Ris->proj(2, 0), mr.Ris->proj(2, 1)});
    auto [covers, a, dep, unit_b] = w.app(facts, {affs[0], affs[1]})->projs<4>();
    if (covers != w.lit_tt()) return w.lit_ff();
    if (plan) *plan = {*bytes, a, dep, unit_b, w.tuple({affs[0], affs[1]})};
    return w.lit_tt();
}

namespace {

// Whether `body`, an application of `⊕`, reassociates exactly or by its math mode and has `init` as its unit.
bool monoid(const Def* body, const Def* init) {
    auto& w   = body->world();
    auto unit = [&](f64 v) {
        auto width = math::isa_f(init->type());
        return width && init == math::lit_f(w, *width, v);
    };
    if (auto ar = Axm::isa<math::arith>(body)) {
        auto mode = Lit::isa(ar->callee()->as<App>()->callee()->as<App>()->arg());
        if (!mode || !(*mode & std::to_underlying(math::Mode::reassoc))) return false;
        return (ar.id() == math::arith::add && unit(0.0)) || (ar.id() == math::arith::mul && unit(1.0));
    }
    if (auto ex = Axm::isa<math::extrema>(body)) {
        bool max = ex.id() == math::extrema::iM || ex.id() == math::extrema::IM;
        return unit(max ? -std::numeric_limits<f64>::infinity() : std::numeric_limits<f64>::infinity());
    }
    if (auto wr = Axm::isa<core::wrap>(body)) {
        if (Lit::isa(wr->callee()->as<App>()->callee()->as<App>()->arg()) != 0) return false;
        return (wr.id() == core::wrap::add && Lit::isa(init) == 0)
            || (wr.id() == core::wrap::mul && Lit::isa(init) == 1);
    }
    return false;
}

// The operation `⊕` that a `tensor.fold_of` combiner keeps, if it is a monoid with unit `init`, or `nullptr`.
const Def* reduce_monoid(const Def* comb, const Def* init) {
    auto& w = comb->world();
    if (Axm::isa<tensor::fold_of, 1>(comb)) {
        auto [op, _] = comb->as<App>()->args<2>();
        return w.app(w.app(w.annex<tensor::monoid_of>(), init->type()), {op, init}) == w.lit_tt() ? op : nullptr;
    }
    return nullptr;
}

} // namespace

bool pe_generic_plan(const App* callee, PeGeneric* plan, bool bounded) {
    auto& w    = callee->world();
    auto mr    = map_red(callee);
    auto ranks = mr.ranks();
    if (!ranks || ranks->ro < 1 || ranks->nis < 1) return false;
    u64 ro = ranks->ro, rn = ranks->nloops(), nis = ranks->nis;
    if (!lit_projs(mr.Ris, nis)) return false;
    for (u64 d = ro; d != rn && !bounded; ++d)
        if (!Lit::isa(mr.Sr->proj(rn, d))) return false;
    if (mr.To == w.type_nat() || !Check::test_alpha(mr.init->type(), mr.To)) return false;

    // The output is a permutation of the parallel loops, which only literal extents can tile.
    auto raw   = btensor::MapRedPost::of(callee);
    auto loops = out_loops(raw.map_out, mr.Rn, mr.Ro);
    if (!loops) return false;
    u64 wdim = ro - 1;
    for (u64 k = 0; k != ro; ++k) {
        auto d = (*loops)[k];
        if (mr.So->proj(ro, k) != mr.Sr->proj(rn, d)) return false;
        if (d != k && !bounded)
            for (u64 e = 0; e != rn; ++e)
                if (!Lit::isa(mr.Sr->proj(rn, e))) return false;
        if (k == ro - 1) wdim = d;
    }

    auto affs = raw.maps->projs(nis);
    for (size_t i = 0; i != nis; ++i)
        if (!(affs[i] = aff_canon(affs[i], mr.Rn, mr.Ris->proj(nis, i)))) return false;
    auto reads = DefVec(nis), units = DefVec(nis);
    for (size_t i = 0; i != nis; ++i) {
        auto R    = mr.Ris->proj(nis, i);
        reads[i]  = w.app(w.app(w.annex<tensor::aff_reads>(), {mr.Rn, R}), affs[i]);
        auto unit = w.app(w.annex<tensor::aff_unit>(), {mr.Rn, R});
        units[i]  = w.tuple(DefVec(rn, [&](size_t d) { return w.app(unit, {affs[i], w.lit_idx(rn, d)}); }));
    }
    auto facts = w.app(w.app(w.annex<tensor::generic_facts>(), {mr.nis, mr.Rn}),
                       {mr.Ro, w.lit_nat(wdim), w.tuple(reads), w.tuple(units)});
    auto [red_unit, out_unit, rdim, outer] = facts->projs<4>();
    if (!Lit::isa(red_unit) || !Lit::isa(out_unit) || !Lit::isa(rdim) || !Lit::isa(outer)) return false;
    auto merge = reduce_monoid(mr.comb, mr.init);
    auto bytes = Lit::isa(w.app(w.annex(core::trait::size), mr.To));
    u64 lanes  = merge && bytes && *bytes != 0 ? btensor::vec_regs(w.driver()).bytes / *bytes : 0;
    if (plan)
        *plan = {w.tuple(affs), red_unit, out_unit, w.lit_nat(wdim), rdim, merge, w.lit_nat(lanes), w.lit_nat(bytes.value_or(0)), outer};
    return true;
}

const Def* aff_canon(const Def* aff, const Def* n, const Def* m) {
    auto& w   = aff->world();
    auto name = aff->isa<Name>();
    auto r    = Lit::isa(m);
    if (!name || !r || name->op()->num_projs() != 3) return aff;
    auto [s, vals, rows] = name->op()->projs<3>();
    auto ns              = Lit::isa(s);
    if (!ns) return aff;
    auto canon = AffCanon(w, n, vals, *ns);
    for (size_t j = 0; j != *r; ++j)
        if (!canon.scan(rows->proj(*r, j))) return nullptr;
    auto new_rows = DefVec(*r, [&](size_t j) { return canon.build(rows->proj(*r, j)); });
    auto map      = w.app(w.annex<tensor::aff_map>(), {n, m, w.lit_nat(canon.syms().size())});
    return w.app(map, {w.tuple(canon.syms()), w.tuple(new_rows)});
}

bool aff_static(const Def* aff) {
    auto name = aff->isa<Name>();
    return name && name->op()->proj(3, 2)->is_closed();
}

std::optional<fe::Vector<u64>> out_loops(const Def* map_out, const Def* Rn, const Def* Ro) {
    auto& w  = Rn->world();
    auto aff = aff_canon(map_out, Rn, Ro);
    auto ro  = Lit::isa(Ro);
    if (!aff || !ro) return {};
    auto [ok, loops] = w.app(w.app(w.annex<tensor::aff_perm>(), {Rn, Ro}), aff)->projs<2>();
    if (ok != w.lit_tt()) return {};
    return lit_projs(loops, *ro);
}

bool aff_identity(const Def* aff, const Def* n, const Def* m) {
    auto rn = Lit::isa(n);
    if (!rn || Lit::isa(m) != rn) return false;
    auto loops = out_loops(aff, n, m);
    if (!loops) return false;
    for (u64 k = 0; k != *rn; ++k)
        if ((*loops)[k] != k) return false;
    return true;
}

bool aff_reads_once(const Def* aff, const Def* n, const Def* m) {
    auto& w = n->world();
    if (!Lit::isa(n) || !Lit::isa(m)) return false;
    auto data = aff_canon(aff, n, m);
    return data && w.app(w.app(w.annex<tensor::aff_injective>(), {n, m}), data) == w.lit_tt();
}

std::optional<u64> aff_last_dim(const Def* aff, const Def* n, const Def* m) {
    auto r    = Lit::isa(m);
    auto data = r && *r != 0 ? aff_canon(aff, n, m) : nullptr;
    auto name = data ? data->isa<Name>() : nullptr;
    auto row  = name ? name->op()->proj(3, 2)->proj(*r, *r - 1)->isa<Name>() : nullptr;
    auto dim  = row ? row->op()->isa<Inj>() : nullptr;
    if (!dim || dim->index() != 0) return {};
    return Lit::isa(dim->value());
}

const Def* aff_lam(const Def* f, const Def* n, const Def* m) {
    auto& w = f->world();
    // `aff_eval` unfolds only static rows.
    if (auto canon = aff_canon(f, n, m)) f = canon;
    return w.app(w.app(w.annex<tensor::aff_eval_map>(), {n, m}), f);
}

btensor::MapRedPost map_red(const App* callee) {
    auto mr  = btensor::MapRedPost::of(callee);
    auto nis = Lit::isa(mr.nis), nps = Lit::isa(mr.nps);
    if (!nis || !nps) return mr;
    auto& w    = callee->world();
    mr.map_out = aff_lam(mr.map_out, mr.Rn, mr.Ro);
    mr.maps    = w.tuple(
        DefVec(*nis, [&](size_t i) { return aff_lam(mr.maps->proj(*nis, i), mr.Rn, mr.Ris->proj(*nis, i)); }));
    mr.post_maps = w.tuple(
        DefVec(*nps, [&](size_t j) { return aff_lam(mr.post_maps->proj(*nps, j), mr.Ro, mr.Rps->proj(*nps, j)); }));
    mr.maps_all = w.tuple({mr.maps, mr.post_maps});
    return mr;
}

// Only a primitive operation states its laws; a function's body is not inspected.
const Def* normalize_monoid_of(const Def*, const Def*, const Def* arg) {
    auto& w         = arg->world();
    auto [op, unit] = arg->projs<2>();
    if (op->isa_mut<Lam>()) return w.lit_ff();
    if (!op->is_closed()) return nullptr;
    auto T     = unit->type();
    auto probe = w.mut_lam(w.sigma({T, T}), T);
    return w.lit_bool(monoid(w.app(op, probe->var()), unit));
}

const Def* normalize_id(const Def*, const Def*, const Def* arg) {
    auto [data, ret] = arg->projs<2>();
    return arg->world().app(ret, data->proj(2, 0));
}

const Def* normalize_copy(const Def*, const Def*, const Def* arg) {
    auto [data, ret] = arg->projs<2>();
    return arg->world().app(ret, data->proj(2, 1));
}

const Def* normalize_map_of(const Def*, const Def* c, const Def* arg) {
    auto& w          = arg->world();
    auto app         = c->as<App>()->arg();
    auto [data, ret] = arg->projs<2>();
    auto xs          = data->proj(2, 1);
    auto lam         = app->isa_mut<Lam>();
    return w.app(ret, lam && lam->is_set() ? lam->reduce_body(xs) : w.app(app, xs));
}

const Def* normalize_fold_of(const Def*, const Def* c, const Def* arg) {
    auto& w          = arg->world();
    auto [op, h]     = c->as<App>()->args<2>();
    auto [data, ret] = arg->projs<2>();
    auto [acc, xs]   = data->projs<2>();
    // Like the body of a combiner, which inlines its operations.
    auto apply = [&](const Def* f, const Def* x) {
        auto lam = f->isa_mut<Lam>();
        return lam && lam->is_set() ? lam->reduce_body(x) : w.app(f, x);
    };
    return w.app(ret, apply(op, w.tuple({acc, apply(h, xs)})));
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

/// A tuple by its elements, since a tuple of equal elements would print as a pack.
std::string show_elems(const Def* def, size_t n) {
    return std::format("({})", fe::Join(DefVec(n, [&](size_t i) { return def->proj(n, i); }), ", "));
}

/// A `tensor.SchedTree` as one indented node per line, e.g. `Split 0, 8, Fixed` above `Stage 0` above `Leaf 1`.
void show_tree(std::string& out, const Def* tree, size_t depth) {
    static constexpr std::array names = {"Leaf", "Split", "Reduce", "Pack", "Stage", "Alloc", "Switch"};
    static constexpr std::array modes = {"Fixed", "Loop", "Peel"};
    auto indent                       = std::string(2 * depth, ' ');
    auto name                         = tree->isa<Name>();
    auto inj                          = name ? name->op()->isa<Inj>() : nullptr;
    if (!inj || inj->index() >= names.size()) {
        out += std::format("{}{}\n", indent, tree);
    } else if (inj->index() == 0) {
        out += std::format("{}Leaf {}\n", indent, inj->value());
    } else {
        auto arg  = inj->value();
        auto n    = arg->num_projs();
        auto kids = inj->index() == names.size() - 1 ? 2_u64 : 1_u64;
        auto args = std::vector<std::string>(n - kids);
        for (size_t i = 0; i != n - kids; ++i) {
            auto field = arg->proj(n, i);
            auto mode  = field->isa<Name>() ? field->as<Name>()->op()->isa<Inj>() : nullptr;
            auto lit   = Lit::isa(field);
            auto elems = field->num_projs();
            args[i]    = mode && mode->index() < modes.size() ? std::string(modes[mode->index()])
                       : lit                                  ? std::to_string(*lit)
                       : elems > 1                            ? show_elems(field, elems)
                                                              : std::format("{}", field);
        }
        out += std::format("{}{} {}\n", indent, names[inj->index()], fe::Join(args, ", "));
        for (auto i = n - kids; i != n; ++i)
            show_tree(out, arg->proj(n, i), depth + 1);
    }
}

std::string show_root(const Def* ro,
                      const Def* lanes,
                      const Def* sizes,
                      const Def* caps,
                      const Def* dyn,
                      const Def* fuse,
                      const Def* reads,
                      size_t rn,
                      size_t nis) {
    auto slots = std::vector<std::string>(nis);
    for (size_t i = 0; i != nis; ++i)
        slots[i] = show_elems(reads->proj(nis, i), rn);
    return std::format("ro {} lanes {} sizes {} caps {} dyn {} fuse {} reads {}", ro, lanes, show_elems(sizes, rn),
                       show_elems(caps, rn), show_elems(dyn, rn), show_elems(fuse, nis), fe::Join(slots, " "));
}

namespace {

/// A `tensor.AffE` in prefix notation, e.g. `add dim 0 mul dim 2 3`; a symbol without a static value stays `sym`.
void show_aff(std::string& out, const Def* e, const Def* vals, size_t s) {
    static constexpr std::array ops = {"dim", "sym", "lit", "add", "mul", "fdiv", "cdiv", "mod", "sub"};
    auto name                       = e->isa<Name>();
    auto inj                        = name ? name->op()->isa<Inj>() : nullptr;
    if (!inj || inj->index() >= ops.size()) {
        out += " ?";
        return;
    }
    auto i = inj->index();
    auto v = inj->value();
    if (i == 1) {
        auto k   = Lit::isa(v);
        auto val = k && *k < s ? Lit::isa(vals->proj(s, *k)) : std::nullopt;
        out += val ? std::format(" lit {}", *val) : std::string(" sym");
    } else if (i < 3) {
        auto l = Lit::isa(v);
        out += l ? std::format(" {} {}", ops[i], *l) : std::string(" ?");
    } else {
        out += std::format(" {}", ops[i]);
        show_aff(out, v->proj(2, 0), vals, s);
        if (i == 3 || i == 8) {
            show_aff(out, v->proj(2, 1), vals, s);
        } else {
            auto k   = v->proj(2, 1)->isa<Name>();
            auto num = k ? k->op()->isa<Inj>() : nullptr;
            auto c   = num && num->index() == 0 ? Lit::isa(num->value()) : std::nullopt;
            out += c ? std::format(" {}", *c) : std::string(" ?");
        }
    }
}

} // namespace

std::string show_maps(const Def* maps, const Def* shapes, const Def* Ris, size_t nis) {
    auto rows = std::string(), dims = std::string();
    for (size_t i = 0; i != nis; ++i) {
        auto r    = Lit::isa(Ris->proj(nis, i));
        auto map  = maps->proj(nis, i);
        auto name = map->isa<Name>();
        auto body = name ? name->op() : map;
        rows += " slot";
        dims += " slot";
        if (!r || body->num_projs() != 3) {
            rows += " row ?";
            continue;
        }
        auto [s, vals, es] = body->projs<3>();
        auto ns            = Lit::isa(s);
        for (size_t j = 0; j != *r; ++j) {
            rows += " row";
            if (ns)
                show_aff(rows, es->proj(*r, j), vals, *ns);
            else
                rows += " ?";
            auto n = Lit::isa(shapes->proj(nis, i)->proj(*r, j));
            dims += n ? std::format(" {}", *n) : std::string(" ?");
        }
    }
    return std::format("rows{} shapes{}", rows, dims);
}

const Def* normalize_sched_show(const Def*, const Def* callee, const Def* arg) {
    auto& w                                 = arg->world();
    auto [N, Rn]                            = callee->as<App>()->args<2>();
    auto [ro, caps, dyn, fuse, reads, tree] = arg->projs<6>();
    auto nis = Lit::isa(N), rn = Lit::isa(Rn);
    if (!nis || !rn) return nullptr;
    for (size_t d = 0; d != *rn; ++d)
        if (!Lit::isa(caps->proj(*rn, d)) || !Lit::isa(dyn->proj(*rn, d))) return nullptr;
    auto name = tree->isa<Name>();
    if (!name || !name->op()->isa<Inj>()) return nullptr;
    auto text = std::string();
    show_tree(text, tree, 0);
    w.log().v("schedule tree: {}\n{}", show_root(ro, w.lit_nat_0(), caps, caps, dyn, fuse, reads, *rn, *nis), text);
    return tree;
}

MIM_tensor_NORMALIZER_IMPL

} // namespace mim::plug::tensor
