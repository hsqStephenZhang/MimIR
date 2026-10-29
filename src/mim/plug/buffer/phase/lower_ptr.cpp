#include "mim/plug/buffer/phase/lower_ptr.h"

#include <mim/axm.h>
#include <mim/def.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/buffer/buffer.h"

namespace mim::plug::buffer {

namespace {

u64 rank_of(const Def* s) {
    if (auto rank = Lit::isa<u64>(s->arity())) return *rank;
    s->blame("cannot lower buffer with a non-literal rank").bail();
}

bool has_runtime_extent(const Def* s, u64 rank) {
    for (u64 i = 0; i != rank; ++i)
        if (!Lit::isa(s->proj(rank, i))) return true;
    return false;
}

// Runtime extents must not survive in pointee types.
const Def* type_ptr(const Def* s, const Def* T) {
    auto& w = s->world();
    if (has_runtime_extent(s, rank_of(s))) s = w.top(w.type_nat());
    return w.call<mem::Ptr>(Defs{w.arr(s, T), w.lit_nat_0()});
}

/// Allocates the precise `«s; T»` (sized by `s`'s product) and hands out the erased handle.
std::pair<const Def*, const Def*> alloc_erased(const Def* s, const Def* T, const Def* mem) {
    auto& w          = s->world();
    auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
    if (auto erased = type_ptr(s, T); erased != ptr->type()) ptr = w.call<core::bitcast>(erased, ptr);
    return {mem2, ptr};
}

// Erased pointers cannot retain multidimensional indices.
const Def* flat_index(const Def* s, const Def* idx) {
    auto r = rank_of(s);
    if (!has_runtime_extent(s, r)) return idx;
    auto& w           = s->world();
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

std::pair<const Def*, const Def*> ptr_elem(const Def* t) {
    if (auto ptr = Axm::isa<mem::Ptr>(t)) {
        auto [elem, space] = ptr->args<2>();
        while (auto array = elem->isa<Arr>())
            elem = array->body();
        return {elem, space};
    }
    return {nullptr, nullptr};
}

bool is_erased(const Def* t) {
    if (auto arr = Axm::isa<mem::Ptr>(t)->arg(2, 0)->isa<Arr>()) return arr->arity()->isa<Top>();
    return false;
}

} // namespace

void LowerPtr::remember_empty(const Def* s, const Def* T, const Def* ptr) {
    for (auto extent : s->projs())
        if (Lit::isa<u64>(extent) == 0) {
            while (auto array = T->isa<Arr>())
                T = array->body();
            empty_elements_[ptr].emplace(T);
            return;
        }
}

const Def* LowerPtr::rewrite_imm_App(const App* app) {
    auto& w = new_world();

    if (auto buf_ax = Axm::isa<buffer::Buf>(app)) {
        auto [r, s, T] = buf_ax->args<3>();
        s              = rewrite(s);
        T              = rewrite(T);
        auto ptr       = type_ptr(s, T);
        remember_empty(s, T, ptr);
        return ptr;
    } else if (auto alloc_ax = Axm::isa<buffer::alloc>(app)) {
        auto mem         = rewrite(alloc_ax->arg());
        auto [r, s, T]   = alloc_ax->callee()->as<App>()->args<3>();
        s                = rewrite(s);
        T                = rewrite(T);
        auto [mem2, ptr] = alloc_erased(s, T, mem);
        remember_empty(s, T, ptr->type());
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
        auto s               = rewrite(copy_ax->callee()->as<App>()->arg(3, 1));
        if (has_runtime_extent(s, rank_of(s)))
            app->blame("cannot lower `buffer.copy` of a runtime extent to a whole-array load/store").bail();
        mem = rewrite(mem);
        dst = rewrite(dst);
        src = rewrite(src);
        // Whole-buffer copy: load the entire array out of `src` and store it into `dst`.
        auto [mem2, val] = w.call<mem::load>(Defs{mem, src})->projs<2>();
        return w.call<mem::store>(Defs{mem2, dst, val});
    } else if (auto free_ax = Axm::isa<buffer::free>(app)) {
        auto [mem, buf] = free_ax->args<2>([this](const Def* d) { return rewrite(d); });
        return w.call<mem::free>(Defs{mem, buf});
    } else if (auto init_ax = Axm::isa<buffer::init>(app)) {
        auto [mem, val] = init_ax->args<2>();
        auto [r, s, T]  = init_ax->callee()->as<App>()->args<3>();
        mem             = rewrite(mem);
        val             = rewrite(val);
        s               = rewrite(s);
        T               = rewrite(T);
        if (has_runtime_extent(s, rank_of(s)))
            app->blame("cannot lower `buffer.init` of a runtime extent to a whole-array store").bail();
        auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
        remember_empty(s, T, ptr->type());
        auto mem3 = w.call<mem::store>(Defs{mem2, ptr, val});
        return w.tuple({mem3, ptr});
    } else if (auto const_ax = Axm::isa<buffer::lit>(app)) {
        auto [mem, val] = const_ax->args<2>();
        auto [r, s, T]  = const_ax->callee()->as<App>()->args<3>();
        mem             = rewrite(mem);
        val             = rewrite(val);
        s               = rewrite(s);
        T               = rewrite(T);
        if (has_runtime_extent(s, rank_of(s)))
            app->blame("cannot lower `buffer.lit` of a runtime extent to a whole-array store").bail();
        auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
        remember_empty(s, T, ptr->type());
        auto mem3 = w.call<mem::store>(Defs{mem2, ptr, w.pack(s, val)});
        return w.tuple({mem3, ptr});
    } else if (auto buf_alloc_copy = Axm::isa<gpu::buf_alloc_copy>(app)) {
        auto s = rewrite(buf_alloc_copy->callee()->as<App>()->arg(3, 1));
        if (has_runtime_extent(s, rank_of(s))) app->blame("cannot lower a device transfer of a runtime extent").bail();
        auto [m0, m1, ptr] = buf_alloc_copy->args<3>([this](const Def* d) { return rewrite(d); });
        return w.call(gpu::alloc_copy::block, Defs{m0, m1, ptr});
    } else if (auto buf_copy_to_host = Axm::isa<gpu::buf_copy_to_host>(app)) {
        auto s = rewrite(buf_copy_to_host->callee()->as<App>()->arg(3, 1));
        if (has_runtime_extent(s, rank_of(s))) app->blame("cannot lower a device transfer of a runtime extent").bail();
        auto [m0, m1, d_ptr, h_ptr] = buf_copy_to_host->args<4>([this](const Def* d) { return rewrite(d); });
        return w.call(gpu::copy_to_host::block, Defs{m0, m1, d_ptr, h_ptr});
    }

    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerPtr::coerce_leaf(const Def* type, const Def* val) {
    if (auto res = RWPhase::coerce_leaf(type, val)) return res;
    // Pre-erasure checking positions justify adapting precise and erased pointers.
    auto [de, da] = ptr_elem(type);
    auto [se, sa] = ptr_elem(val->type());
    if (de && se && da == sa && (is_erased(type) || is_erased(val->type()))) {
        auto compatible = de == se;
        if (auto i = empty_elements_.find(type); i != empty_elements_.end()) compatible |= i->second.contains(se);
        if (auto i = empty_elements_.find(val->type()); i != empty_elements_.end())
            compatible |= i->second.contains(de);
        if (compatible) return type->world().call<core::bitcast>(type, val);
    }
    return nullptr;
}

} // namespace mim::plug::buffer
