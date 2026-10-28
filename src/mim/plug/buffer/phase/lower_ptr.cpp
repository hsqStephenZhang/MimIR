#include "mim/plug/buffer/phase/lower_ptr.h"

#include <mim/axm.h>
#include <mim/def.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/buffer/buffer.h"

namespace mim::plug::buffer {

namespace {

/// Does the shape @p s have a runtime extent (a non-literal dim)?
bool has_runtime_extent(const Def* s) {
    auto r = Lit::isa(s->arity());
    if (!r) return false;
    for (size_t i = 0; i != *r; ++i)
        if (!Lit::isa(s->proj(*r, i))) return true;
    return false;
}

/// The pointer a buffer handle lowers to.
/// A runtime extent is erased: the pointee is the flat `«⊤; T»`, so no type below this phase names the extent, and
/// the element address is linearized from the logical shape (a value next to the handle, see `flat_index`).
const Def* type_ptr(const Def* s, const Def* T) {
    auto& w = s->world();
    if (has_runtime_extent(s)) s = w.top(w.type_nat());
    return w.call<mem::Ptr>(Defs{w.arr(s, T), w.lit_nat_0()});
}

/// Allocates the precise `«s; T»` (sized by `s`'s product) and hands out the erased handle.
std::pair<const Def*, const Def*> alloc_erased(const Def* s, const Def* T, const Def* mem) {
    auto& w          = s->world();
    auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
    if (auto erased = type_ptr(s, T); erased != ptr->type()) ptr = w.call<core::bitcast>(erased, ptr);
    return {mem2, ptr};
}

/// The index tuple to `lea` with: for an erased handle the row-major linearization `Σ idx#k · ∏_{j>k} s#j` as one
/// `Idx ⊤`, computed on the `Idx 0` carrier like affine.linearize.
const Def* flat_index(const Def* s, const Def* idx) {
    if (!has_runtime_extent(s)) return idx;
    auto& w           = s->world();
    auto r            = *Lit::isa(s->arity());
    auto i64          = w.type_i64();
    const Def* lin    = w.lit(i64, 0);
    const Def* stride = w.lit_nat(1);
    for (size_t k = r; k-- != 0;) {
        auto x    = w.call(core::conv::u, w.lit_nat_0(), idx->proj(r, k));
        auto term = w.call(core::wrap::mul, core::Mode::none, Defs{x, w.call<core::bitcast>(i64, stride)});
        lin       = w.call(core::wrap::add, core::Mode::none, Defs{lin, term});
        stride    = w.call(core::nat::mul, Defs{stride, s->proj(r, k)});
    }
    return w.call<core::bitcast>(w.type_idx(w.top(w.type_nat())), lin);
}

/// Is @p t the pointer to an erased (`«⊤; T»`) or precise array?
bool is_arr_ptr(const Def* t) {
    if (auto ptr = Axm::isa<mem::Ptr>(t)) return ptr->arg(2, 0)->isa<Arr>();
    return false;
}

/// Adapts @p v to the type @p dst where the two differ only by erasure, component-wise through sigmas.
/// A pointer is cast; precise→erased forgets the extent, erased→precise reinstates the one the caller chose.
/// A continuation is contravariant: an η-wrapper takes @p dst's parameters and adapts them to @p v's.
const Def* adapt(const Def* dst, const Def* v) {
    auto& w = dst->world();
    if (v->type() == dst) return v;
    if (auto sig = dst->isa<Sigma>(); sig && !sig->isa_mut()) {
        auto n = sig->num_ops();
        if (v->type()->isa<Sigma>() && v->type()->num_ops() == n)
            return w.tuple(DefVec(n, [&](size_t i) { return adapt(sig->op(i), v->proj(n, i)); }));
    }
    if (auto pi = Pi::isa_cn(dst))
        if (auto vpi = Pi::isa_cn(v->type()); vpi && vpi->dom()->num_projs() == pi->dom()->num_projs()) {
            auto k = w.mut_con(pi->dom())->set(v->dbg());
            k->set(true, w.app(v, adapt(vpi->dom(), k->var())));
            return k;
        }
    if (is_arr_ptr(dst) && is_arr_ptr(v->type())) return w.call<core::bitcast>(dst, v);
    return v;
}

} // namespace

const Def* LowerPtr::rewrite_imm_App(const App* app) {
    auto& w = new_world();

    if (auto buf_ax = Axm::isa<buffer::Buf>(app)) {
        auto [r, s, T] = buf_ax->args<3>();
        s              = rewrite(s);
        T              = rewrite(T);
        return type_ptr(s, T);
    } else if (auto alloc_ax = Axm::isa<buffer::alloc>(app)) {
        auto mem         = rewrite(alloc_ax->arg());
        auto [r, s, T]   = alloc_ax->callee()->as<App>()->args<3>();
        s                = rewrite(s);
        T                = rewrite(T);
        auto [mem2, ptr] = alloc_erased(s, T, mem);
        return w.tuple({mem2, ptr});
    } else if (auto read_ax = Axm::isa<buffer::read>(app)) {
        auto [mem, buf, idx] = read_ax->args<3>();
        auto s               = rewrite(read_ax->callee()->as<App>()->arg(3, 1));
        mem                  = rewrite(mem);
        buf                  = rewrite(buf);
        idx                  = flat_index(s, rewrite(idx));
        auto element_ptr     = mem::op_lea_tuple(buf, idx);
        auto [mem2, val]     = w.call<mem::load>(Defs{mem, element_ptr})->projs<2>();
        return w.tuple({mem2, val});
    } else if (auto write_ax = Axm::isa<buffer::write>(app)) {
        auto [mem, buf, idx, val] = write_ax->args<4>();
        auto s                    = rewrite(write_ax->callee()->as<App>()->arg(3, 1));
        mem                       = rewrite(mem);
        buf                       = rewrite(buf);
        idx                       = flat_index(s, rewrite(idx));
        val                       = rewrite(val);
        auto element_ptr          = mem::op_lea_tuple(buf, idx);
        auto mem2                 = w.call<mem::store>(Defs{mem, element_ptr, val});
        return w.tuple({mem2, buf});
    } else if (auto copy_ax = Axm::isa<buffer::copy>(app)) {
        auto [mem, dst, src] = copy_ax->args<3>();
        if (has_runtime_extent(rewrite(copy_ax->callee()->as<App>()->arg(3, 1))))
            app->blame("cannot lower `buffer.copy` of a runtime extent to a whole-array load/store").bail();
        mem                  = rewrite(mem);
        dst                  = rewrite(dst);
        src                  = rewrite(src);
        // Whole-buffer copy: load the entire array out of `src` and store it into `dst`.
        auto [mem2, val] = w.call<mem::load>(Defs{mem, src})->projs<2>();
        return w.call<mem::store>(Defs{mem2, dst, val});
    } else if (auto free_ax = Axm::isa<buffer::free>(app)) {
        auto [mem, buf] = free_ax->args<2>([this](const Def* d) { return rewrite(d); });
        return w.call<mem::free>(Defs{mem, buf});
    } else if (auto init_ax = Axm::isa<buffer::init>(app)) {
        auto [mem, val]  = init_ax->args<2>();
        auto [r, s, T]   = init_ax->callee()->as<App>()->args<3>();
        mem              = rewrite(mem);
        val              = rewrite(val);
        s                = rewrite(s);
        T                = rewrite(T);
        if (has_runtime_extent(s))
            app->blame("cannot lower `buffer.init` of a runtime extent to a whole-array store").bail();
        auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
        auto mem3        = w.call<mem::store>(Defs{mem2, ptr, val});
        return w.tuple({mem3, ptr});
    } else if (auto const_ax = Axm::isa<buffer::lit>(app)) {
        auto [mem, val]  = const_ax->args<2>();
        auto [r, s, T]   = const_ax->callee()->as<App>()->args<3>();
        mem              = rewrite(mem);
        val              = rewrite(val);
        s                = rewrite(s);
        T                = rewrite(T);
        if (has_runtime_extent(s))
            app->blame("cannot lower `buffer.lit` of a runtime extent to a whole-array store").bail();
        auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
        auto mem3        = w.call<mem::store>(Defs{mem2, ptr, w.pack(s, val)});
        return w.tuple({mem3, ptr});
    } else if (auto buf_alloc_copy = Axm::isa<gpu::buf_alloc_copy>(app)) {
        auto [m0, m1, ptr] = buf_alloc_copy->args<3>([this](const Def* d) { return rewrite(d); });
        return w.call(gpu::alloc_copy::block, Defs{m0, m1, ptr});
    } else if (auto buf_copy_to_host = Axm::isa<gpu::buf_copy_to_host>(app)) {
        auto [m0, m1, d_ptr, h_ptr] = buf_copy_to_host->args<4>([this](const Def* d) { return rewrite(d); });
        return w.call(gpu::copy_to_host::block, Defs{m0, m1, d_ptr, h_ptr});
    }

    // A callee whose buffer parameter has a symbolic shape takes the erased pointer and hands an erased result to
    // its continuation; a precise argument and a precise continuation are adapted to it.
    if (auto pi = app->callee()->type()->isa<Pi>()) {
        auto callee = rewrite(app->callee());
        auto arg    = rewrite(app->arg());
        return w.app(callee, adapt(callee->type()->as<Pi>()->dom(), arg));
    }

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::buffer
