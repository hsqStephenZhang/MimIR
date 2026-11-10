#include "mim/plug/tensor/phase/panel_layout.h"

#include <algorithm>
#include <bit>
#include <numeric>

#include <mim/world.h>

#include <mim/plug/btensor/btensor.h>
#include <mim/plug/tensor/tensor.h>

namespace mim::plug::tensor::panel {

bool Layout::identity() const {
    if (split != 1 || swizzle) return false;
    for (size_t p = 0; p != perm.size(); ++p)
        if (perm[p] != p) return false;
    return true;
}

namespace {

// The dims of a panel after a layout's split and permutation; a split dim is merged back into the second
// physical dim, so the buffer keeps the panel's rank.
std::vector<u64> permuted_dims(const std::vector<u64>& dims, const Layout& layout) {
    auto pre = dims;
    if (layout.split != 1) {
        pre.back() /= layout.split;
        pre.push_back(layout.split);
    }
    auto result = std::vector<u64>(pre.size());
    for (size_t p = 0; p != pre.size(); ++p)
        result[p] = pre[layout.perm[p]];
    return result;
}

size_t merged_dims(size_t rank, const Layout& layout) { return layout.perm.size() - rank + 1; }

std::vector<u64> physical_dims(const std::vector<u64>& dims, const Layout& layout) {
    auto permuted = permuted_dims(dims, layout);
    auto merged   = merged_dims(dims.size(), layout);
    auto result   = std::vector<u64>{permuted[0], 1};
    for (size_t p = 1; p != 1 + merged; ++p)
        result[1] *= permuted[p];
    result.insert(result.end(), permuted.begin() + 1 + merged, permuted.end());
    return result;
}

// Folds @p coords (over the permuted dims) with @p mul and @p add into the physical rank.
template<class T, class Mul, class Add>
std::vector<T>
merge_coords(const std::vector<T>& coords, const std::vector<u64>& permuted, size_t rank, Mul mul, Add add) {
    auto merged = permuted.size() - rank + 1;
    auto result = std::vector<T>{coords[0]};
    auto acc    = coords[1];
    for (size_t p = 2; p != 1 + merged; ++p)
        acc = add(mul(acc, permuted[p]), coords[p]);
    result.push_back(acc);
    result.insert(result.end(), coords.begin() + 1 + merged, coords.end());
    return result;
}

std::vector<u64> strides_of(const std::vector<u64>& shape) {
    auto strides = std::vector<u64>(shape.size(), 1);
    for (size_t d = shape.size() - 1; d-- > 0;)
        strides[d] = strides[d + 1] * shape[d + 1];
    return strides;
}

/// The fill of a panel with literal @p dims, moving `vec` elements along dim `inner` per step, fastest last.
struct Fill {
    Fill(const std::vector<u64>& dims, size_t inner, u64 vec, u64 width)
        : n(dims.size())
        , inner(inner)
        , vec(vec)
        , width(width)
        , chunked(dims) {
        chunked[inner] /= vec;
        for (size_t d = 0; d != n; ++d)
            if (d != inner) order.push_back(d);
        order.push_back(inner);
        for (auto c : chunked)
            total *= c;
    }

    /// The physical offset of logical panel coordinates under a layout and the padded physical shape.
    u64 offset(std::vector<u64> coords, const Layout& layout, const std::vector<u64>& shape) const {
        if (layout.split != 1) {
            auto last     = coords.back();
            coords.back() = last / layout.split;
            coords.push_back(last % layout.split);
        }
        auto permuted = std::vector<u64>(layout.perm.size());
        for (size_t p = 0; p != permuted.size(); ++p)
            permuted[p] = coords[layout.perm[p]];
        auto physical = merge_coords(
            permuted, layout.permuted, n, [](u64 a, u64 k) { return a * k; }, [](u64 a, u64 b) { return a + b; });
        auto strides = strides_of(shape);
        u64 row = physical[0], col = 0;
        for (size_t p = 1; p != shape.size(); ++p)
            col += physical[p] * strides[p];
        if (auto& swizzle = layout.swizzle) {
            auto group = (row >> swizzle->shift) % swizzle->groups;
            auto chunk
                = swizzle->xored ? (col / swizzle->width) ^ group : (col / swizzle->width + group) % swizzle->chunks;
            col = chunk * swizzle->width + col % swizzle->width;
        }
        return row * strides[0] + col;
    }

    /// The coordinates of element `v` of fill step `l`.
    std::vector<u64> coords(u64 l, u64 v) const {
        auto coords = std::vector<u64>(n);
        u64 rest    = l;
        for (auto d : order | std::views::reverse) {
            coords[d] = rest % chunked[d];
            rest /= chunked[d];
        }
        coords[inner] = coords[inner] * vec + v;
        return coords;
    }

    /// Whether every fill step's vector stays one aligned run in the layout.
    bool contiguous(const Layout& layout, const std::vector<u64>& shape) const {
        for (u64 l = 0; l != std::min<u64>(total, 64); ++l) {
            auto first = offset(coords(l, 0), layout, shape);
            if (first % width) return false;
            for (u64 v = 1; v != vec; ++v)
                if (offset(coords(l, v), layout, shape) != first + v) return false;
        }
        return true;
    }

    size_t n, inner;
    u64 vec, width;
    std::vector<u64> chunked;
    std::vector<size_t> order;
    u64 total = 1;
};

} // namespace

const Def* term(const Layout& layout, const Def* logical, const std::vector<u64>& physical) {
    auto& w       = logical->world();
    auto rank     = logical->num_projs();
    auto extended = layout.perm;
    if (layout.split == 1) extended.insert(extended.begin() + 2, rank);
    auto perm    = w.tuple(DefVec(rank + 1, [&](size_t p) { return w.lit_idx(rank + 1, extended[p]); }));
    auto swizzle = layout.swizzle.value_or(Swizzle{1, 1, 0, 1});
    auto make    = [&](const Def* pad) {
        return w.app(w.app(w.annex(tensor::layout_expr::make), w.lit_nat(rank)),
                     {logical, w.lit_nat(layout.split), perm, pad, w.lit_nat(swizzle.width), w.lit_nat(swizzle.groups),
                      w.lit_nat(swizzle.shift), w.lit_nat(swizzle.chunks), w.lit_bool(swizzle.xored)});
    };
    auto base_expr = make(w.tuple(DefVec(rank, [&](size_t) { return w.lit_nat_0(); })));
    if (physical.empty()) return base_expr;
    auto base = w.app(w.annex(tensor::layout_expr::shape), {base_expr, logical});
    auto pad  = w.tuple(DefVec(rank, [&](size_t p) { return w.lit_nat(physical[p] - Lit::as(base->proj(rank, p))); }));
    return make(pad);
}

// A tensor-core panel keeps its source's contiguous dim innermost, so its fill copies 16-byte runs.
// The pad makes ldmatrix's 8 rows of a narrow panel, or the scalar fragment loads of a wide one, hit distinct banks.
Choice mma(const Def* storage, size_t inner) {
    auto& w   = storage->world();
    auto n    = storage->num_projs();
    auto dims = std::vector<u64>(n);
    for (size_t d = 0; d != n; ++d)
        dims[d] = Lit::as(storage->proj(n, d));
    auto layout = Layout{std::vector<size_t>()};
    for (size_t d = 0; d != n; ++d)
        if (d != inner) layout.perm.push_back(d);
    layout.perm.push_back(inner);
    layout.permuted = permuted_dims(dims, layout);
    auto shape      = physical_dims(dims, layout);
    auto& last      = shape.back();
    if (last <= 32 && last / 4 % 2 == 0)
        last += 4;
    else if (last > 32 && last % 32 == 0)
        last += 8;
    auto expr     = term(layout, storage, shape);
    auto physical = w.app(w.annex(tensor::layout_expr::shape), {expr, storage});
    return {expr, physical, layout, true};
}

// The storage of a panel whose fill moves `vec` elements of `bytes` along dim `inner` per step, in the layout that
// spreads the fill's warp stores and the items' warp reads over the 32 shared-memory banks best, and whether the
// fill's vector stays contiguous in it.
Choice choose(const Def* storage, size_t inner, u64 vec, u64 bytes, const Reads* reads) {
    auto& w    = storage->world();
    auto n     = storage->num_projs();
    auto dims  = std::vector<u64>(n);
    auto plain = Layout{std::vector<size_t>(n)};
    for (size_t p = 0; p != n; ++p)
        plain.perm[p] = p;
    for (size_t d = 0; d != n; ++d) {
        auto lit = Lit::isa(storage->proj(n, d));
        if (!lit || !*lit) return {nullptr, storage, plain, inner + 1 == n};
        dims[d] = *lit;
    }
    plain.permuted = dims;
    if (n < 2 || bytes < 4 || bytes % 4 || 16 % bytes) return {nullptr, storage, plain, inner + 1 == n};
    auto words = bytes / 4, width = 16 / bytes;
    auto fill  = Fill(dims, inner, vec, width);
    auto total = fill.total;
    // The issue slot plus the shared-memory wavefronts of one warp-wide access per `lanes` lane, each thread
    // touching `per_thread` words.
    auto wavefronts = [&](u64 lanes, u64 per_thread, auto address) {
        u64 phase = std::max<u64>(1, 32 / per_thread), sum = lanes;
        for (u64 lane = 0; lane != lanes; ++lane)
            for (u64 first = 0; first < 32; first += phase) {
                auto banks = std::array<std::vector<u64>, 32>();
                for (u64 t = first; t != first + phase; ++t) {
                    auto word = address(t, lane);
                    if (!word) continue;
                    for (u64 k = 0; k != per_thread; ++k) {
                        auto& bank = banks[(*word + k) % 32];
                        if (std::ranges::find(bank, *word + k) == bank.end()) bank.push_back(*word + k);
                    }
                }
                u64 worst = 1;
                for (auto& bank : banks)
                    worst = std::max<u64>(worst, bank.size());
                sum += worst;
            }
        return sum;
    };
    auto cost = [&](const Layout& layout, const std::vector<u64>& shape) {
        auto vector_store = vec != 1 && fill.contiguous(layout, shape);
        u64 warps = std::clamp<u64>((total + 31) / 32, 1, 8), sum = 0;
        for (u64 warp = 0; warp != warps; ++warp)
            sum += wavefronts(vector_store ? 1 : vec, (vector_store ? vec : 1) * words,
                              [&](u64 t, u64 v) -> std::optional<u64> {
                                  auto l = warp * 32 + t;
                                  if (l >= total) return {};
                                  return fill.offset(fill.coords(l, v), layout, shape) * words;
                              });
        if (!reads) return sum;
        // Every item reads its whole dims at the tile its id selects, for each value of the fixed dims.
        auto fixed = std::vector<size_t>();
        u64 steps = 1, items = 1;
        for (size_t d = 0; d != n; ++d)
            if (!reads->loop_dim[d] && !reads->whole[d]) fixed.push_back(d), steps *= dims[d];
        auto strides = strides_of(reads->counts);
        for (auto c : reads->counts)
            items *= c;
        u64 per_item = 1;
        for (size_t d = 0; d != n; ++d)
            if (reads->whole[d]) per_item *= dims[d];
        auto item_coords = [&](u64 id, u64 step, u64 e) {
            auto coords = std::vector<u64>(n);
            for (auto d : fixed | std::views::reverse)
                coords[d] = step % dims[d], step /= dims[d];
            for (size_t d = n; d-- > 0;)
                if (reads->whole[d]) coords[d] = e % dims[d], e /= dims[d];
            for (size_t d = 0; d != n; ++d)
                if (auto j = reads->loop_dim[d]) coords[d] = (id / strides[*j]) % reads->counts[*j];
            return coords;
        };
        // An item's elements that lie in one aligned 16-byte run load as one vector.
        auto chunks = std::vector<std::pair<u64, u64>>();
        for (u64 e = 0; e < per_item;) {
            auto first = fill.offset(item_coords(0, 0, e), layout, shape);
            u64 run    = 1;
            while (shape.back() % width == 0 && first % width == 0 && run != width && e + run < per_item
                   && fill.offset(item_coords(0, 0, e + run), layout, shape) == first + run)
                ++run;
            if (run != width) run = 1;
            chunks.emplace_back(e, run);
            e += run;
        }
        u64 item_warps = std::clamp<u64>((items + 31) / 32, 1, 8);
        for (u64 step = 0; step != steps; ++step)
            for (u64 warp = 0; warp != item_warps; ++warp)
                for (auto [e, run] : chunks)
                    sum += wavefronts(1, run * words, [&](u64 t, u64) -> std::optional<u64> {
                        auto id = warp * 32 + t;
                        if (id >= items) return {};
                        return fill.offset(item_coords(id, step, e), layout, shape) * words;
                    });
        return sum;
    };
    auto best        = cost(plain, dims);
    auto best_layout = plain;
    auto best_shape  = dims;
    auto consider    = [&](const Layout& layout, const std::vector<u64>& shape) {
        if (auto c = cost(layout, shape); c < best) best = c, best_layout = layout, best_shape = shape;
    };
    for (auto split : {u64(1), width})
        if (split == 1 || (dims.back() % split == 0 && dims.back() > split)) {
            auto pre  = n + (split != 1);
            auto perm = std::vector<size_t>(pre);
            for (size_t p = 0; p != pre; ++p)
                perm[p] = p;
            do {
                if (split != 1 && perm.back() != pre - 1) continue;
                auto layout     = Layout{perm, split, {}};
                layout.permuted = permuted_dims(dims, layout);
                auto shape      = physical_dims(dims, layout);
                consider(layout, shape);
                for (size_t p = 1; p != shape.size(); ++p) {
                    u64 tail = 1;
                    for (size_t q = p + 1; q < shape.size(); ++q)
                        tail *= shape[q];
                    for (u64 pad = 1; pad != 8; ++pad) {
                        auto padded = shape;
                        padded[p] += pad;
                        if (padded[p] * tail * bytes % 16) continue;
                        consider(layout, padded);
                    }
                }
                u64 cols = 1;
                for (size_t p = 1; p != shape.size(); ++p)
                    cols *= shape[p];
                // A swizzle moves whole chunks, so it needs the chunk as the last dim: a lane then stays a literal.
                // At equal cost an exclusive or wins: its chunk index needs no addition.
                if (std::has_single_bit(cols) && cols / width >= 2 && shape.back() == width)
                    for (auto xored : {true, false})
                        for (u64 groups = 2; groups <= std::min<u64>(cols / width, 8); groups *= 2)
                            for (u64 shift = 0; shift != 4; ++shift) {
                                auto swizzled    = layout;
                                swizzled.swizzle = Swizzle{width, groups, shift, cols / width, xored};
                                consider(swizzled, shape);
                            }
            } while (std::next_permutation(perm.begin(), perm.end()));
        }
    if (!best_layout.identity() || best_shape != dims)
        w.log().d("panel {} stored as {} split={} perm={} swizzle={}", storage, fe::Join(best_shape), best_layout.split,
                  fe::Join(best_layout.perm),
                  best_layout.swizzle
                      ? std::format("{} (row >> {}) % {}", best_layout.swizzle->xored ? "xor" : "rotate",
                                    best_layout.swizzle->shift, best_layout.swizzle->groups)
                      : "none");
    auto expr     = term(best_layout, storage, best_shape);
    auto physical = w.app(w.annex(tensor::layout_expr::shape), {expr, storage});
    return {expr, physical, best_layout, vec != 1 && fill.contiguous(best_layout, best_shape)};
}

const Def* compose_read_map(const Def* inner, const Def* outer) {
    auto& w  = outer->world();
    auto pi  = outer->type()->as<Pi>();
    auto map = w.mut_lam(pi->dom(), inner->type()->as<Pi>()->codom());
    return map->set(true, w.app(inner, w.app(outer, map->var())));
}

std::optional<PureRead> tensor_read(const Def* value, const Def* shape) {
    value      = Axm::peel<tensor::buf>(value);
    auto& w    = value->world();
    auto adapt = [&](std::optional<PureRead> read, const Def* produced) {
        if (read && produced != shape) {
            auto rank   = shape->arity();
            auto bridge = w.app(w.annex<tensor::reshape_map>(), {produced->arity(), rank});
            bridge      = w.app(bridge, {produced, shape});
            read->map   = compose_read_map(read->map, bridge);
        }
        return read;
    };
    if (auto slice = Axm::isa<tensor::slice>(value)) {
        auto [meta, shape, box]    = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, step, extent] = box->projs<3>();
        auto [type, rank]          = meta->projs<2>();
        auto map                   = w.app(w.app(w.annex<tensor::slice_map>(), rank), {start, step});
        return adapt(PureRead{slice->arg(), map, type, rank, shape}, extent);
    }
    if (auto slice = Axm::isa<tensor::slice_box>(value)) {
        auto [meta, shape, box]   = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, extent, cap] = box->projs<3>();
        auto [type, rank]         = meta->projs<2>();
        auto map = w.app(w.app(w.annex<tensor::slice_map>(), rank), {start, w.pack(rank, w.lit_nat_1())});
        return adapt(PureRead{slice->arg(), map, type, rank, shape}, extent);
    }
    auto read = is_pure_read(value);
    return read ? adapt(read, btensor::MapRedPost::of(value->as<App>()->callee()->as<App>()).So) : read;
}

// The dim of a rank-`r` index that `read` sends unscaled to the last physical coordinate, if any,
// with the gcd of the coordinate's other terms (0 if there are none).
std::optional<std::pair<size_t, u64>> contiguous_dim(const Def* read, const Def* rank, const Def* physical_rank) {
    auto& old = read->world();
    auto r = Lit::isa(rank), ri = Lit::isa(physical_rank);
    if (!r || !ri || !*r || !*ri) return {};
    auto probe = old.mut_lam(read->type()->as<Pi>());
    auto lam   = read->isa<Lam>();
    auto out   = lam && lam->is_set() ? lam->reduce(probe->var()).back() : old.app(read, probe->var());
    auto last  = Linear::of(out->proj(*ri, *ri - 1));
    auto dim   = std::optional<size_t>();
    u64 rest   = std::abs(last.constant);
    for (auto [atom, coefficient] : last.atoms) {
        auto probed = std::optional<size_t>();
        for (size_t d = 0; d != *r; ++d)
            if (atom == probe->var(*r, d)) probed = d;
        if (probed && coefficient == 1) {
            if (dim) return {};
            dim = probed;
        } else {
            rest = std::gcd(rest, u64(std::abs(coefficient)));
        }
    }
    if (!dim) return {};
    return std::pair{*dim, rest};
}

Choice decode(const Def* expr, const Def* storage, size_t inner, u64 vec, u64 bytes) {
    auto& w   = storage->world();
    auto n    = storage->num_projs();
    auto dims = std::vector<u64>(n);
    for (size_t d = 0; d != n; ++d)
        dims[d] = Lit::as(storage->proj(n, d));
    auto data = expr;
    while (auto name = data->isa<Name>())
        data = name->op();
    auto fields  = data->projs<9>();
    auto layout  = Layout{};
    layout.split = Lit::as(fields[1]);
    for (size_t p = 0; p != n + 1; ++p)
        if (layout.split != 1 || p != 2) layout.perm.push_back(Lit::as(fields[2]->proj(n + 1, p)));
    auto swizzle = Swizzle{Lit::as(fields[4]), Lit::as(fields[5]), Lit::as(fields[6]), Lit::as(fields[7]),
                           Lit::as(fields[8]) != 0};
    if (swizzle.width != 1 || swizzle.groups != 1 || swizzle.shift != 0 || swizzle.chunks != 1)
        layout.swizzle = swizzle;
    layout.permuted = permuted_dims(dims, layout);
    auto physical   = w.app(w.annex(tensor::layout_expr::shape), {expr, storage});
    auto shape      = std::vector<u64>(n);
    for (size_t d = 0; d != n; ++d)
        shape[d] = Lit::as(physical->proj(n, d));
    auto contiguous = bytes >= 4 && bytes % 4 == 0 && 16 % bytes == 0 && vec != 1
                   && Fill(dims, inner, vec, 16 / bytes).contiguous(layout, shape);
    return {expr, physical, layout, contiguous};
}

} // namespace mim::plug::tensor::panel
