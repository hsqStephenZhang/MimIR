#include <algorithm>
#include <array>
#include <bit>
#include <optional>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#elif defined(__linux__)
#    include <unistd.h>
#endif

#include <fe/format.h>

#include <mim/def.h>
#include <mim/driver.h>
#include <mim/plugin.h>
#include <mim/tuple.h>
#include <mim/world.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/core/core.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

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
    // Size-1 axes collapse out of the type, so equal types suffice.
    auto [Tr, s_in, s_out] = c->as<App>()->uncurry_args<3>();
    if (s_in == s_out || arg->type() == type) return arg;
    return nullptr;
}

const Def* normalize_reshape(const Def* type, const Def* c, const Def* arg) {
    // Identity reshape: if the input and output shapes agree, the reshape is a no-op.
    // Size-1 axes collapse out of the type, so equal types suffice.
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

/// What `tensor.gemm_tiles` derives its blocks from.
struct Target {
    u64 vec_bytes, vec_regs, l1;
    bool lane_fma; ///< Whether a vector FMA takes its broadcast operand from a lane, which costs a register per row.
    u64 bufs;      ///< Operand sets in flight per K step: 2 on an in-order core that pipelines the loads by hand.
};

std::optional<u64> sysctl_u64([[maybe_unused]] const char* name) {
#ifdef __APPLE__
    u64 v    = 0;
    size_t n = sizeof(v);
    if (sysctlbyname(name, &v, &n, nullptr, 0) == 0 && v != 0) return v;
#endif
    return {};
}

Target host_target() {
    Target t{16, 16, 32 * 1024, false, 1};
#if defined(__aarch64__) || defined(_M_ARM64)
    t.vec_regs = 32, t.lane_fma = true;
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    if (__builtin_cpu_supports("avx512f"))
        t.vec_bytes = 64, t.vec_regs = 32;
    else if (__builtin_cpu_supports("avx2"))
        t.vec_bytes = 32;
#endif
#ifdef __APPLE__
    // A hybrid CPU reports its performance cores as level 0.
    if (auto v = sysctl_u64("hw.perflevel0.l1dcachesize").or_else([] { return sysctl_u64("hw.l1dcachesize"); }))
        t.l1 = *v;
#elif defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    if (auto v = sysconf(_SC_LEVEL1_DCACHE_SIZE); v > 0) t.l1 = u64(v);
#endif
    return t;
}

/// The host's target with the `-X tensor:` overrides of @p d applied.
Target target(Driver& d) {
    static const auto host = host_target();
    auto t                 = host;
    if (auto v = d.arg_nat("tensor", "vec-bytes")) t.vec_bytes = *v;
    if (auto v = d.arg_nat("tensor", "vec-regs")) t.vec_regs = *v;
    if (auto v = d.arg_nat("tensor", "l1")) t.l1 = *v;
    if (auto v = d.arg_nat("tensor", "lane-fma")) t.lane_fma = *v != 0;
    if (auto v = d.arg_nat("tensor", "bufs")) t.bufs = std::max<u64>(1, *v);
    return t;
}

u64 pow2_floor(u64 x) { return x == 0 ? 1 : std::bit_floor(x); }
u64 ceil_div(u64 a, u64 b) { return (a + b - 1) / b; }

/// The largest register block height whose accumulators, B row and A operands fit the vector registers.
u64 max_mr(const Target& t, u64 nv) {
    auto regs = [&](u64 mr) { return mr * nv + t.bufs * nv + t.bufs * (t.lane_fma ? mr : 1); };
    u64 mr    = 1;
    while (regs(mr + 1) <= t.vec_regs) ++mr;
    return mr;
}

/// The block height up to @p hi for @p M rows: fewest blocks, then the most even ones.
/// Below `hi / 2` rows it stays `hi`, which leaves such a product to the untiled schedule.
u64 fit_mr(u64 hi, u64 M) {
    auto lo = std::max<u64>(1, ceil_div(hi, 2));
    if (M < lo) return hi;
    auto best = hi;
    for (auto mr = hi; mr >= lo; --mr) {
        auto cost = [&](u64 m) { return std::pair{ceil_div(M, m), ceil_div(M, m) * m - M}; };
        if (cost(mr) < cost(best)) best = mr;
    }
    return best;
}

} // namespace

const Def* normalize_gemm_tiles(const Def*, const Def* callee, const Def* arg) {
    // An open element type waits for its instance; a closed one without a size, such as `Nat`, takes 8 bytes.
    auto T = callee->as<App>()->arg();
    if (T->is_open()) return nullptr;
    auto [Md, Nd, Kd] = arg->projs<3>();
    auto M            = Lit::isa(Md);
    auto N            = Lit::isa(Nd);
    auto K            = Lit::isa(Kd);
    if (!M || !N || !K) return nullptr;

    auto& w = T->world();
    auto& d = w.driver();
    auto eb = Lit::isa<u64>(w.call(core::trait::size, T)).value_or(8);
    if (eb == 0) eb = 8;
    auto t = target(d);

    // A row of the register block takes `nv` vector registers of accumulators, and the block's B row the same.
    auto lanes = std::max<u64>(1, t.vec_bytes / eb);
    auto nv    = u64(t.vec_regs >= 32 ? 4 : 2);
    auto mr    = d.arg_nat("tensor", "gemm-mr").value_or(fit_mr(max_mr(t, nv), *M));
    auto nr    = d.arg_nat("tensor", "gemm-nr").value_or(nv * lanes);

    // A is read in place, so B's block «kc, nc» shares L1 with it and takes half; four of its panels set the K block.
    auto budget = t.l1 / 2;
    u64 kc;
    if (auto v = d.arg_nat("tensor", "gemm-kc"))
        kc = *K <= *v + *v / 2 ? *K : *v;
    else
        kc = ceil_div(*K, ceil_div(*K, std::clamp<u64>(pow2_floor(budget / (4 * nr * eb)), 64, 512)));
    auto panels = std::min(pow2_floor(std::clamp<u64>(budget / (kc * nr * eb), 1, 8)), pow2_floor(ceil_div(*N, nr)));
    auto nc     = d.arg_nat("tensor", "gemm-nc").value_or(nr * panels);
    return w.tuple({w.lit_nat(mr), w.lit_nat(nr), w.lit_nat(kc), w.lit_nat(nc)});
}

const Def* normalize_shape(const Def*, const Def* c, const Def* arg) {
    // `tensor.shape r arr` is the first `r` axes of `arr`'s (fused) array type.
    auto& w = c->world();
    auto r  = Lit::isa<u64>(c->as<App>()->arg()); // the explicit rank `r`
    if (!r) return nullptr;
    if (*r == 0) return w.tuple();

    auto arr = arg->type()->isa<Arr>();
    if (!arr) return nullptr;
    auto shape = arr->shape();
    auto rank  = shape.rank();
    if (!rank || *rank < *r) return nullptr; // `arr` is not (statically) at least rank `r`
    return *shape.take(*r);
}

MIM_tensor_NORMALIZER_IMPL

} // namespace mim::plug::tensor
