#include "mim/plug/buffer/phase/lower_ptr.h"

#include <mim/axm.h>
#include <mim/def.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/buffer/buffer.h"

namespace mim::plug::buffer {

const Def* LowerPtr::rewrite_imm_App(const App* app) {
    auto& w = new_world();

    if (auto buf_ax = Axm::isa<buffer::Buf>(app)) {
        auto [r, s, T] = buf_ax->args<3>();
        s              = rewrite(s);
        T              = rewrite(T);
        return w.call<mem::Ptr>(Defs{w.arr(s, T), w.lit_nat_0()});
    } else if (auto alloc_ax = Axm::isa<buffer::alloc>(app)) {
        auto mem         = rewrite(alloc_ax->arg());
        auto [r, s, T]   = alloc_ax->callee()->as<App>()->args<3>();
        s                = rewrite(s);
        T                = rewrite(T);
        auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
        return w.tuple({mem2, ptr});
    } else if (auto read_ax = Axm::isa<buffer::read>(app)) {
        auto [mem, buf, idx] = read_ax->args<3>();
        mem                  = rewrite(mem);
        buf                  = rewrite(buf);
        idx                  = rewrite(idx);
        auto element_ptr     = mem::op_lea_tuple(buf, idx);
        auto [mem2, val]     = w.call<mem::load>(Defs{mem, element_ptr})->projs<2>();
        return w.tuple({mem2, val});
    } else if (auto vectorize_ax = Axm::isa<buffer::vectorize>(app)) {
        auto [n, c]    = vectorize_ax->callee()->as<App>()->args<2>([this](const Def* d) { return rewrite(d); });
        auto [r, s, T] = vectorize_ax->callee()->as<App>()->callee()->as<App>()->args<3>();
        T              = rewrite(T);
        auto vectors   = w.call<mem::Ptr>(Defs{w.arr(c, w.arr(n, T)), w.lit_nat_0()});
        auto bytes     = w.call(core::nat::mul, Defs{n, core::op(core::trait::size, T)});
        return w.call<mem::align>(Defs{w.call<core::bitcast>(vectors, rewrite(vectorize_ax->arg())), bytes});
    } else if (auto write_ax = Axm::isa<buffer::write>(app)) {
        auto [mem, buf, idx, val] = write_ax->args<4>();
        mem                       = rewrite(mem);
        buf                       = rewrite(buf);
        idx                       = rewrite(idx);
        val                       = rewrite(val);
        auto element_ptr          = mem::op_lea_tuple(buf, idx);
        auto mem2                 = w.call<mem::store>(Defs{mem, element_ptr, val});
        return w.tuple({mem2, buf});
    } else if (auto copy_ax = Axm::isa<buffer::copy>(app)) {
        auto [mem, dst, src] = copy_ax->args<3>();
        mem                  = rewrite(mem);
        dst                  = rewrite(dst);
        src                  = rewrite(src);
        // Whole-buffer copy: load the entire array out of `src` and store it into `dst`.
        auto [mem2, val] = w.call<mem::load>(Defs{mem, src})->projs<2>();
        return w.call<mem::store>(Defs{mem2, dst, val});
    } else if (auto free_ax = Axm::isa<buffer::free>(app)) {
        auto [mem, buf] = free_ax->args<2>([this](const Def* d) { return rewrite(d); });
        return w.call<mem::free>(Defs{mem, buf});
    } else if (auto persist = Axm::isa<buffer::persist>(app)) {
        auto [meta, source, args] = persist->uncurry_args<3>();
        auto [r, s, T]            = meta->projs<3>();
        auto [mem, src]           = rewrite(args)->projs<2>();
        return w.app(w.app(w.annex<mem::persist>(), {w.arr(rewrite(s), rewrite(T)), rewrite(source)}), {mem, src});
    } else if (auto init_ax = Axm::isa<buffer::init>(app)) {
        auto [mem, val]  = init_ax->args<2>();
        auto [r, s, T]   = init_ax->callee()->as<App>()->args<3>();
        mem              = rewrite(mem);
        val              = rewrite(val);
        s                = rewrite(s);
        T                = rewrite(T);
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
        auto [mem2, ptr] = mem::op_alloc(w.arr(s, T), mem)->projs<2>();
        auto mem3        = w.call<mem::store>(Defs{mem2, ptr, w.pack(s, val)});
        return w.tuple({mem3, ptr});
    } else if (auto ld = Axm::isa<gpu::buf_ldmatrix>(app)) {
        auto [n, mem, buf, idx] = ld->args<4>([this](const Def* d) { return rewrite(d); });
        auto row                = mem::op_lea_tuple(buf, idx);
        return w.app(w.app(w.annex<gpu::ldmatrix>(), w.lit_nat_0()), {n, mem, row});
    } else if (auto map = Axm::isa<gpu::buf_tensor_map>(app)) {
        auto [buf, box_rows, box_cols] = map->args<3>([this](const Def* d) { return rewrite(d); });
        auto T                         = rewrite(map->decurry()->decurry()->arg());
        auto s                         = rewrite(map->decurry()->arg());
        auto base                      = rewrite(map->arg(3, 0));
        auto [X, a]                    = Axm::as<mem::Ptr>(base->type())->args<2>();
        return w.app(w.app(w.app(w.app(w.annex<gpu::device_tensor_map>(), T), X), a),
                     {base, s->proj(2, 0), s->proj(2, 1), box_rows, box_cols});
    } else if (auto addr = Axm::isa<gpu::buf_addr>(app)) {
        auto [buf, idx] = addr->args<2>([this](const Def* d) { return rewrite(d); });
        auto ptr        = mem::op_lea_tuple(buf, idx);
        auto T          = Axm::as<mem::Ptr>(ptr->type())->arg(2, 0);
        return w.app(w.app(w.app(w.annex<gpu::shared_addr>(), T), w.lit_nat_0()), ptr);
    } else if (auto ld = Axm::isa<gpu::buf_ldmatrix_b16>(app)) {
        auto [n, trans, mem, buf, idx] = ld->args<5>([this](const Def* d) { return rewrite(d); });
        auto T                         = rewrite(ld->decurry()->decurry()->arg());
        auto row                       = mem::op_lea_tuple(buf, idx);
        return w.app(w.app(w.app(w.annex<gpu::ldmatrix_b16>(), T), w.lit_nat_0()), {n, trans, mem, row});
    } else if (auto buf_alloc_copy = Axm::isa<gpu::buf_alloc_copy>(app)) {
        auto [m0, m1, ptr] = buf_alloc_copy->args<3>([this](const Def* d) { return rewrite(d); });
        return w.call(gpu::alloc_copy::block, Defs{m0, m1, ptr});
    } else if (auto buf_copy_to_host = Axm::isa<gpu::buf_copy_to_host>(app)) {
        auto [m0, m1, d_ptr, h_ptr] = buf_copy_to_host->args<4>([this](const Def* d) { return rewrite(d); });
        return w.call(gpu::copy_to_host::block, Defs{m0, m1, d_ptr, h_ptr});
    }

    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::buffer
