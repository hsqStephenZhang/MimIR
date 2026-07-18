#include "mim/plug/ll_nvptx/phase/ll_nvptx.h"

#include <format>

#include <mim/driver.h>

#include <mim/util/sys.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/ll_nvptx/ll_nvptx.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

using namespace std::string_literals;

namespace mim::plug::ll_nvptx {

namespace core = mim::plug::core;
namespace ll   = mim::plug::ll;
namespace math = mim::plug::math;
namespace mem  = mim::plug::mem;
namespace gpu  = mim::plug::gpu;

class HostEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream, std::optional<std::string> device_fatbin_file)
        : Super(world, "llvm_nvptx_host_emitter", ostream)
        , device_fatbin_file_(device_fatbin_file) {}

    void start() final;
    void find_kernels(const Def*);

    void emit_epilogue(Lam*) final;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

protected:
    std::string convert(const Def*, bool simd = true) override;

private:
    static constexpr std::string_view mod_name_          = "@.mimir_cu_mod";
    static constexpr std::string_view ctx_name_          = "@.mimir_cu_ctx";
    static constexpr std::string_view fatbin_name_       = "@.fatbin";
    static constexpr std::string_view kernel_array_name_ = "@.mimir_kernels";
    static constexpr std::string_view kernel_name_prefix = "@.kname.";

    void emit_cu_error_handling(ll::BB&, const std::string&, bool at_tail = false);

    std::optional<std::string> device_fatbin_file_;
    LamMap<int> kernel_ids_;
    absl::btree_map<std::string, int> kernel_name2id_;
    absl::btree_map<int, std::string> kernel_id2name_;

    DefSet analyzed_;
};

class DeviceEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_device_emitter", ostream)
        , uses_libdevice(false) {}

    void start() final;

    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

    bool is_using_libdevice() const { return uses_libdevice; }
    const std::string& get_extra_flags() const { return extra_flags; }

private:
    std::string convert(const Def* def, bool simd = false) override;

    absl::btree_map<std::string, int> symbols_;
    LamSet kernels_;

    bool uses_libdevice;
    std::string extra_flags;
};

void HostEmitter::start() {
    for (auto def : world().annexes().defs())
        find_kernels(def);
    for (auto def : world().externals().muts())
        find_kernels(def);

    for (auto [kid, name] : kernel_id2name_) {
        std::print(vars_decls_, "{}{} = private constant [{} x i8] c\"{}\\00\"\n", kernel_name_prefix, kid,
                   name.size() + 1, name);
    }
    std::print(vars_decls_, "{} = dso_local global [{} x ptr] zeroinitializer\n", kernel_array_name_,
               kernel_id2name_.size());

    Super::start();
}

void HostEmitter::find_kernels(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    for (auto d : def->deps())
        find_kernels(d);

    auto register_kernel = [&](auto launch) {
        auto kernel     = launch->decurry()->decurry()->arg();
        auto kernel_lam = kernel->template isa_mut<Lam>();
        assert(kernel_lam && "Expect kernel passed to %gpu.launch to be a mutable lambda");
        if (kernel_ids_.contains(kernel_lam)) return;
        auto name = id(kernel_lam).substr(1);
        auto kid  = kernel_name2id_.size();
        if (auto i = kernel_name2id_.find(name); i != kernel_name2id_.end())
            kid = i->second;
        else {
            kernel_name2id_[name] = kid;
            kernel_id2name_[kid]  = name;
        }
        kernel_ids_[kernel_lam] = kid;
    };

    if (auto launch = Axm::isa<gpu::launch>(def))
        register_kernel(launch);
    else if (auto launch = Axm::isa<gpu::launch_3d>(def))
        register_kernel(launch);
}

constexpr auto CU_INIT                = "cuInit";
constexpr auto CU_CTX_CREATE          = "cuCtxCreate_v4";
constexpr auto CU_CTX_DESTROY         = "cuCtxDestroy_v2";
constexpr auto CU_DEVICE_GET          = "cuDeviceGet";
constexpr auto CU_LAUNCH_KERNEL       = "cuLaunchKernel_ptsz";
constexpr auto CU_MEM_ALLOC           = "cuMemAlloc_v2";
constexpr auto CU_MEM_ALLOC_ASYNC     = "cuMemAllocAsync_ptsz";
constexpr auto CU_MEM_FREE            = "cuMemFree_v2";
constexpr auto CU_MEM_FREE_ASYNC      = "cuMemFreeAsync_ptsz";
constexpr auto CU_MEMCPY_HTOD         = "cuMemcpyHtoD_v2";
constexpr auto CU_MEMCPY_HTOD_ASYNC   = "cuMemcpyHtoDAsync_v2_ptsz";
constexpr auto CU_MEMCPY_DTOH         = "cuMemcpyDtoH_v2";
constexpr auto CU_MEMCPY_DTOH_ASYNC   = "cuMemcpyDtoHAsync_v2_ptsz";
constexpr auto CU_MODULE_LOAD_FATBIN  = "cuModuleLoadFatBinary";
constexpr auto CU_MODULE_GET_FUNCTION = "cuModuleGetFunction";
constexpr auto CU_MODULE_UNLOAD       = "cuModuleUnload";
constexpr auto CU_STREAM_CREATE       = "cuStreamCreate";
constexpr auto CU_STREAM_DESTROY      = "cuStreamDestroy_v2";
constexpr auto CU_STREAM_SYNC         = "cuStreamSynchronize_ptsz";

void HostEmitter::emit_cu_error_handling(ll::BB& bb, const std::string& cu_result, bool tail) {
    // TODO: implement
    return;
}

std::string HostEmitter::convert(const Def* type, bool simd) {
    if (auto ptr = Axm::isa<mem::Ptr>(type)) {
        auto [_, addr_space] = ptr->args<2>();
        auto lit             = Lit::isa(addr_space);
        if (lit.value_or(0L) != 0) {
            // NVIDIA treats all device pointers as i64s in host code
            return "i64";
        }
    }
    return Super::convert(type, simd);
}

void HostEmitter::emit_epilogue(Lam* lam) {
    auto& bb = lam2bb_[lam];

    // HACK: we partially re-implement the checks in Super::emit_epilogue to catch target-specific applications
    auto app = lam->body()->as<App>();
    if (auto ret = isa_targetspecific_intrinsic(bb, app)) {
        assert(ret.has_value());
        if (app->callee() == root()->ret_var()) // return
            assert(false && "Return not implemented in NVPTX backend");
        else if (auto dispatch = Dispatch(app))
            assert(false && "Dispatch not implemented in NVPTX backend");
        else if (app->callee()->isa<Bot>())
            assert(false && "Bot not implemented in NVPTX backend");
        else if (auto _ = Lam::isa_mut_basicblock(app->callee())) // ordinary jump
            assert(false && "Ordinary Jump not implemented in NVPTX backend");
        else if (Pi::isa_returning(app->callee_type())) // function call
            bb.tail("br label {}", ret.value());
        else
            assert(false && "Unexpected return case in NVPTX backend");
    } else {
        Super::emit_epilogue(lam);
    }
}

std::optional<std::string> HostEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);
    std::string op;

    if (auto default_stream = Axm::isa<gpu::default_stream>(def)) {
        return "null";
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        auto dev_num   = 0; // TODO: consider parameterizing this
        auto ctx_flags = 0; // TODO: consider parameterizing this

        declare("i32 @{}(i32)", CU_INIT);
        auto init_res = bb.assign(name + "_init_res", "call i32 @{}(i32 0)", CU_INIT);
        emit_cu_error_handling(bb, init_res);

        declare("i32 @{}(ptr, i32)", CU_DEVICE_GET);
        auto dev_ptr = bb.assign(name + "_dev_ptr", "alloca i32");
        auto dev_get_res
            = bb.assign(name + "_get_res", "call i32 @{}(ptr {}, i32 {})", CU_DEVICE_GET, dev_ptr, dev_num);
        emit_cu_error_handling(bb, dev_get_res);

        declare("i32 @{}(ptr, ptr, i32, i32)", CU_CTX_CREATE);
        std::print(vars_decls_, "{} = global ptr null\n", ctx_name_);
        auto dev     = bb.assign(name + "_dev", "load i32, ptr {}", dev_ptr);
        auto ctx_res = bb.assign(name + "_ctx_res", "call i32 @{}(ptr {}, ptr null, i32 {}, i32 {})", CU_CTX_CREATE,
                                 ctx_name_, ctx_flags, dev);
        emit_cu_error_handling(bb, ctx_res);

        declare("i32 @{}(ptr, ptr)", CU_MODULE_LOAD_FATBIN);
        std::print(vars_decls_, "{} = global ptr null\n", mod_name_);
        if (device_fatbin_file_.has_value()) {
            std::ifstream fatbin_file(device_fatbin_file_.value(), std::ios::binary);
            if (!fatbin_file) error("Could not open {} as binary file", device_fatbin_file_.value());

            auto start = std::istreambuf_iterator<char>(fatbin_file);
            auto end   = std::istreambuf_iterator<char>();
            std::vector<u8> fatbin_bytes(start, end);

            std::print(vars_decls_, "{} = private constant [{} x i8] c\"", fatbin_name_, fatbin_bytes.size());
            for (auto byte : fatbin_bytes) {
                bool invalid_cstr_char = byte == '"' || byte == '\\';
                if (std::isprint(byte) && !invalid_cstr_char) {
                    std::print(vars_decls_, "{:c}", byte);
                } else {
                    auto byte_val = static_cast<int>(byte);
                    std::print(vars_decls_, "\\{:x}{:x}", byte_val / 16, byte_val % 16);
                }
            }
            std::print(vars_decls_, "\"\n");
        } else {
            std::print(vars_decls_, "; Add the bytes of your compiled nvptx fatbin binary here:\n");
            std::print(vars_decls_,
                       "{} = private constant [YOUR_FATBIN_DATA_SIZE_GOES_HERE x i8] YOUR_FATBIN_DATA_GOES_HERE\n",
                       fatbin_name_);
        }
        auto mod_res = bb.assign(name + "_mod_res", "call i32 @{}(ptr {}, ptr {})", CU_MODULE_LOAD_FATBIN, mod_name_,
                                 fatbin_name_);
        emit_cu_error_handling(bb, mod_res);
        auto mod_inner = bb.assign(name + "_mod_inner", "load ptr, ptr {}", mod_name_);

        declare("i32 @{}(ptr, ptr, ptr)", CU_MODULE_GET_FUNCTION);
        for (auto [kid, kname] : kernel_id2name_) {
            auto func_ptr = bb.assign("%" + kname + "_funcptr", "getelementptr inbounds ptr, ptr {}, i64 {}",
                                      kernel_array_name_, kid);
            auto func_res = bb.assign("%" + kname + "_getfuncres", "call i32 @{}(ptr {}, ptr {}, ptr {}{})",
                                      CU_MODULE_GET_FUNCTION, func_ptr, mod_inner, kernel_name_prefix, kid);
            emit_cu_error_handling(bb, func_res);
        }

        auto mem = init->arg();
        return emit_unsafe(mem);
    } else if (auto deinit = Axm::isa<gpu::deinit>(def)) {
        declare("i32 @{}(ptr)", CU_MODULE_UNLOAD);
        bb.tail("{}_mod = load ptr, ptr {}", name, mod_name_);
        bb.tail("{}_mod_unload_res = call i32 @{}(ptr {}_mod)", name, CU_MODULE_UNLOAD, name);
        emit_cu_error_handling(bb, name + "_mod_unload_res", true);

        declare("i32 @{}(ptr)", CU_CTX_DESTROY);
        bb.tail("{}_ctx = load ptr, ptr {}", name, ctx_name_);
        bb.tail("{}_ctx_destroy_res = call i32 @{}(ptr {}_ctx)", name, CU_CTX_DESTROY, name);
        emit_cu_error_handling(bb, name + "_ctx_destroy_res", true);

        emit_unsafe(deinit->arg(0));
        return emit_unsafe(deinit->arg(1));
    } else if (auto stream_init = Axm::isa<gpu::stream_init>(def)) {
        declare("i32 @{}(ptr, i32)", CU_STREAM_CREATE);

        emit_unsafe(stream_init->arg(0));
        emit_unsafe(stream_init->arg(1));
        auto stream_ptr = emit(stream_init->arg(2));

        auto res = bb.assign(name, "call i32 @{}(ptr {}, i32 0)", CU_STREAM_CREATE, stream_ptr);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        declare("i32 @{}(ptr)", CU_STREAM_DESTROY);

        emit_unsafe(stream_deinit->arg(0));
        emit_unsafe(stream_deinit->arg(1));
        auto stream = emit(stream_deinit->arg(2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", CU_STREAM_DESTROY, stream);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare("i32 @{}(ptr)", CU_STREAM_SYNC);

        emit_unsafe(stream_sync->arg(0));
        emit_unsafe(stream_sync->arg(1));
        auto stream = emit(stream_sync->arg(2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", CU_STREAM_SYNC, stream);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        bool is_async;
        switch (alloc.id()) {
            case gpu::alloc::block: is_async = false; break;
            case gpu::alloc::asyn: is_async = true; break;
            default: fe::unreachable();
        }

        if (is_async)
            declare("i32 @{}(ptr, i64, ptr)", CU_MEM_ALLOC_ASYNC);
        else
            declare("i32 @{}(ptr, i64)", CU_MEM_ALLOC);

        emit_unsafe(alloc->arg(0));
        auto alloc_t    = alloc->decurry()->arg();
        World& w        = alloc_t->world();
        auto type_size  = w.call(core::trait::size, alloc_t);
        auto alloc_size = emit(type_size);

        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));

        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        std::string alloc_res;
        if (is_async) {
            auto stream = emit(alloc->arg(1));
            alloc_res   = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {}, ptr {})", CU_MEM_ALLOC_ASYNC, alloc_ptr,
                                    alloc_size, stream);
        } else
            alloc_res = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {})", CU_MEM_ALLOC, alloc_ptr, alloc_size);

        emit_cu_error_handling(bb, alloc_res);
        return bb.assign(name, "load {}, {} addrspace(0)* {}", ptr_t, ptr_t, alloc_ptr);
    } else if (auto free = Axm::isa<gpu::free>(def)) {
        bool is_async;
        switch (free.id()) {
            case gpu::free::block: is_async = false; break;
            case gpu::free::asyn: is_async = true; break;
            default: fe::unreachable();
        }

        if (is_async)
            declare("i32 @{}(i64)", CU_MEM_FREE_ASYNC);
        else
            declare("i32 @{}(i64)", CU_MEM_FREE);

        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));

        std::string free_res;
        if (is_async) {
            auto stream = emit(free->arg(2));
            free_res    = bb.assign(name + "res", "call i32 @{}(i64 {}, ptr {})", CU_MEM_FREE_ASYNC, ptr, stream);
        } else
            free_res = bb.assign(name + "res", "call i32 @{}(i64 {})", CU_MEM_FREE, ptr);

        emit_cu_error_handling(bb, free_res);
        return free_res;
    } else if (auto copy_to_device = Axm::isa<gpu::copy_to_device>(def)) {
        bool is_async;
        switch (copy_to_device.id()) {
            case gpu::copy_to_device::block: is_async = false; break;
            case gpu::copy_to_device::asyn: is_async = true; break;
            default: fe::unreachable();
        }

        if (is_async)
            declare("i32 @{}(i64, ptr, i64, ptr)", CU_MEMCPY_HTOD_ASYNC);
        else
            declare("i32 @{}(i64, ptr, i64)", CU_MEMCPY_HTOD);

        auto type      = copy_to_device->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_device->arg(0));
        emit_unsafe(copy_to_device->arg(1));
        auto host_ptr = emit(copy_to_device->arg(2));
        auto dev_ptr  = emit(copy_to_device->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_device->arg(4));
            copy_res    = bb.assign(name + "res", "call i32 @{}(i64 {}, ptr {}, i64 {}, ptr {})", CU_MEMCPY_HTOD_ASYNC,
                                    dev_ptr, host_ptr, size, stream);
        } else
            copy_res = bb.assign(name + "res", "call i32 @{}(i64 {}, ptr {}, i64 {})", CU_MEMCPY_HTOD, dev_ptr,
                                 host_ptr, size);

        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto copy_to_host = Axm::isa<gpu::copy_to_host>(def)) {
        bool is_async;
        switch (copy_to_host.id()) {
            case gpu::copy_to_host::block: is_async = false; break;
            case gpu::copy_to_host::asyn: is_async = true; break;
            default: fe::unreachable();
        }
        if (is_async)
            declare("i32 @{}(ptr, i64, i64, ptr)", CU_MEMCPY_DTOH_ASYNC);
        else
            declare("i32 @{}(ptr, i64, i64)", CU_MEMCPY_DTOH);

        auto [type]    = copy_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_host->arg(0));
        emit_unsafe(copy_to_host->arg(1));
        auto dev_ptr  = emit(copy_to_host->arg(2));
        auto host_ptr = emit(copy_to_host->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_host->arg(4));
            copy_res    = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {}, i64 {}, ptr {})", CU_MEMCPY_DTOH_ASYNC,
                                    host_ptr, dev_ptr, size, stream);
        } else
            copy_res = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {}, i64 {})", CU_MEMCPY_DTOH, host_ptr,
                                 dev_ptr, size);

        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        // TODO: rewrite to use modern cuLaunchKernelEx instead
        declare("i32 @{}(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)", CU_LAUNCH_KERNEL);

        auto [implicits, launch_config, kernel_def, arg_def, func_args] = launch->uncurry_args<5>();
        auto [n_groups_def, n_items_def, stream_def, m, MT]             = launch_config->projs<5>();
        auto [mem, ret_lam_def]                                         = func_args->projs<2>();

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) error("kernel is not a lamda {}", kernel_def);
        if (!kernel_ids_.contains(lam)) error("unknown kernel {}", lam);
        auto kid = kernel_ids_[lam];

        auto shared_mem_bytes = 0;
        if (auto smem_count = Lit::as(m)) {
            if (smem_count != 1) error("You can only have one dynamic allocation of shared memory per kernel");
            shared_mem_bytes = Lit::as(world().call(core::trait::size, MT));
        }

        emit_unsafe(mem);
        auto n_groups = emit(n_groups_def);
        auto n_items  = emit(n_items_def);
        auto stream   = emit(stream_def);
        auto kernel   = emit(kernel_def);
        auto arg      = emit(arg_def);
        auto arg_type = convert(arg_def->type());
        auto ret_lam  = emit(ret_lam_def);

        auto func_ptr = bb.assign(name + "_kernptr", "getelementptr inbounds [{} x ptr], [{} x ptr]* {}, i64 0, i64 {}",
                                  kernel_id2name_.size(), kernel_id2name_.size(), kernel_array_name_, kid);
        auto func_inner = bb.assign(name + "_kernel", "load ptr, ptr {}", func_ptr);

        auto arg_wrap = bb.assign(name + "_arg_wrap", "alloca {}", arg_type);
        std::print(bb.body().emplace_back(), "store {} {}, ptr {}", arg_type, arg, arg_wrap);

        auto args_ptr = bb.assign(name + "_args_ptr", "alloca [1 x ptr]");
        std::print(bb.body().emplace_back(), "store ptr {}, ptr {}", arg_wrap, args_ptr);
        auto args_inner
            = bb.assign(name + "_args_inner", "getelementptr inbounds [1 x ptr], ptr {}, i64 0, i64 0", args_ptr);
        auto launch_res
            = bb.assign(name,
                        "call i32 @{}(ptr {}, i32 {}, i32 1, i32 1, i32 {}, i32 1, i32 1, "
                        "i32 {}, ptr {}, ptr {}, ptr null)",
                        CU_LAUNCH_KERNEL, func_inner, n_groups, n_items, shared_mem_bytes, stream, args_inner);
        emit_cu_error_handling(bb, launch_res);
        return ret_lam;
    } else if (auto launch = Axm::isa<gpu::launch_3d>(def)) {
        // TODO: rewrite to use modern cuLaunchKernelEx instead
        declare("i32 @{}(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)", CU_LAUNCH_KERNEL);

        auto [implicits, launch_config, kernel_def, arg_def, func_args] = launch->uncurry_args<5>();
        auto [grid_x_def, grid_y_def, grid_z_def, block_x_def, block_y_def, block_z_def, stream_def, m, MT]
            = launch_config->projs<9>();
        auto [mem, ret_lam_def] = func_args->projs<2>();

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) error("kernel is not a lamda {}", kernel_def);
        if (!kernel_ids_.contains(lam)) error("unknown kernel {}", lam);
        auto kid = kernel_ids_[lam];

        auto shared_mem_bytes = 0;
        if (auto smem_count = Lit::as(m)) {
            if (smem_count != 1) error("You can only have one dynamic allocation of shared memory per kernel");
            shared_mem_bytes = Lit::as(world().call(core::trait::size, MT));
        }

        emit_unsafe(mem);
        auto grid_x  = emit(grid_x_def);
        auto grid_y  = emit(grid_y_def);
        auto grid_z  = emit(grid_z_def);
        auto block_x = emit(block_x_def);
        auto block_y = emit(block_y_def);
        auto block_z = emit(block_z_def);
        auto stream  = emit(stream_def);
        auto kernel  = emit(kernel_def);
        auto arg     = emit(arg_def);
        auto arg_type = convert(arg_def->type());
        auto ret_lam = emit(ret_lam_def);

        auto func_ptr = bb.assign(name + "_kernptr", "getelementptr inbounds [{} x ptr], [{} x ptr]* {}, i64 0, i64 {}",
                                  kernel_id2name_.size(), kernel_id2name_.size(), kernel_array_name_, kid);
        auto func_inner = bb.assign(name + "_kernel", "load ptr, ptr {}", func_ptr);

        auto arg_wrap = bb.assign(name + "_arg_wrap", "alloca {}", arg_type);
        std::print(bb.body().emplace_back(), "store {} {}, ptr {}", arg_type, arg, arg_wrap);

        auto args_ptr = bb.assign(name + "_args_ptr", "alloca [1 x ptr]");
        std::print(bb.body().emplace_back(), "store ptr {}, ptr {}", arg_wrap, args_ptr);
        auto args_inner
            = bb.assign(name + "_args_inner", "getelementptr inbounds [1 x ptr], ptr {}, i64 0, i64 0", args_ptr);
        auto launch_res
            = bb.assign(name,
                        "call i32 @{}(ptr {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, "
                        "i32 {}, ptr {}, ptr {}, ptr null)",
                        CU_LAUNCH_KERNEL, func_inner, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                        shared_mem_bytes, stream, args_inner);
        emit_cu_error_handling(bb, launch_res);
        return ret_lam;
    }
    return std::nullopt;
}

void DeviceEmitter::start() {
    for (auto kernel : world().externals().muts()) {
        auto kernel_lam = kernel->isa_mut<Lam>();
        assert(kernel_lam && "Expect kernel passed to %gpu.launch to be a mutable lambda");
        kernels_.emplace(kernel_lam);
    }
    Super::start();
    return;
}

std::string DeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    std::print(func_impls_, "define ptx_kernel {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    const Def* smem = nullptr;
    const Def* arg  = nullptr;
    if (kernel->num_vars() == 9) {
        auto [m1, m3, m4, m5, group_id, item_id, smem_1d, arg_1d, ret_lam] = kernel->vars<9>();
        smem = smem_1d;
        arg  = arg_1d;
    } else if (kernel->num_vars() == 13) {
        auto [m1, m3, m4, m5, block_x, block_y, block_z, thread_x, thread_y, thread_z, smem_3d, arg_3d, ret_lam]
            = kernel->vars<13>();
        smem = smem_3d;
        arg  = arg_3d;
    } else {
        error("kernel '{}' has unsupported GPU entry arity {}", kernel, kernel->num_vars());
    }

    auto arg_name = id(arg);
    locals_[arg]  = arg_name;
    std::print(func_impls_, "{} {}) {{\n", convert(arg->type()), arg_name);

    auto& bb = lam2bb_[kernel];

    auto register_sreg_idx = [&](const Def* def, std::string_view sreg) {
        auto name        = id(def);
        auto type        = def->type();
        auto type_name   = convert(type);
        auto opt_idx_lit = Idx::isa_lit(type);
        if (!opt_idx_lit) error("Type of '{}' must have known index type but has {}", def, type);
        auto idx_lit = opt_idx_lit.value();
        locals_[def] = name;
        declare("i32 @llvm.nvvm.read.ptx.sreg.{}()", sreg);
        if (type_name == "i0") {
            locals_[def] = "0";
        } else if (type_name == "i32") {
            bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.{}()", sreg);
        } else if (idx_lit < (1u << 31)) {
            auto i32 = bb.assign(name + "i32", "call i32 @llvm.nvvm.read.ptx.sreg.{}()", sreg);
            bb.assign(name, "trunc i32 {} to {}", i32, type_name);
        } else {
            error("Warp ID too large, must fit into I32");
        }
    };

    if (kernel->num_vars() == 9) {
        auto [m1, m3, m4, m5, group_id, item_id, smem_1d, arg_1d, ret_lam] = kernel->vars<9>();
        register_sreg_idx(group_id, "ctaid.x");
        register_sreg_idx(item_id, "tid.x");
    } else {
        auto [m1, m3, m4, m5, block_x, block_y, block_z, thread_x, thread_y, thread_z, smem_3d, arg_3d, ret_lam]
            = kernel->vars<13>();
        register_sreg_idx(block_x, "ctaid.x");
        register_sreg_idx(block_y, "ctaid.y");
        register_sreg_idx(block_z, "ctaid.z");
        register_sreg_idx(thread_x, "tid.x");
        register_sreg_idx(thread_y, "tid.y");
        register_sreg_idx(thread_z, "tid.z");
    }

    auto shared_as = Lit::as(world().annex<gpu::addr_space_shared>());
    if (auto sigma = smem->type()->isa<Sigma>()) {
        assert(sigma->num_ops() == 0 && "Expect empty sigma for shared memory variable");
    } else {
        auto ptr = Axm::isa<mem::Ptr>(smem->type());
        assert(ptr && "Expect pointer type for shared memory variable");
        auto [T, a] = ptr->args<2>();
        assert(Lit::as(a) == shared_as && "Expect shared memory pointer type for shared memory variable");
        auto name     = "@" + smem->unique_name();
        locals_[smem] = name;
        std::print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", name, a, convert(T));
    }

    return kernel->unique_name();
}

std::string DeviceEmitter::convert(const Def* def, bool simd) {
    if (auto ptr = Axm::isa<mem::Ptr>(def)) {
        auto [T, addr_space] = ptr->args<2>();
        auto local_as        = Lit::as(world().annex<gpu::addr_space_local>());
        if (Lit::as(addr_space) == local_as) return std::format("{}*", Super::convert(T, false));
        if (auto arr = T->isa<Arr>(); arr && math::match_f32(arr->body())) {
            if (auto arity = Lit::isa(arr->arity()); arity && (*arity == 2 || *arity == 4))
                return std::format("<{} x float> addrspace({})*", *arity, Lit::as(addr_space));
        }
    }
    return Super::convert(def, simd);
}

std::optional<std::string> DeviceEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    auto shared_as = Lit::as(world().annex<gpu::addr_space_shared>());
    auto local_as  = Lit::as(world().annex<gpu::addr_space_local>());

    auto simd_f32_arr = [](const Def* type) -> std::optional<nat_t> {
        auto arr = type->isa<Arr>();
        if (!arr || !math::match_f32(arr->body())) return std::nullopt;
        auto arity = Lit::isa(arr->arity());
        if (!arity || (*arity != 2 && *arity != 4)) return std::nullopt;
        return *arity;
    };

    auto simd_ptr_type = [&](const Def* ptr_type) -> std::optional<std::string> {
        auto ptr = Axm::isa<mem::Ptr>(ptr_type);
        if (!ptr) return std::nullopt;
        auto [pointee, addr_space] = ptr->args<2>();
        auto width = simd_f32_arr(pointee);
        if (!width) return std::nullopt;
        return std::format("<{} x float> addrspace({})*", *width, Lit::as(addr_space));
    };

    auto has_contract = [](const Def* mode) {
        auto m = static_cast<math::Mode>(Lit::as(mode));
        return m == math::Mode::fast || fe::has_flag(m, math::Mode::contract);
    };

    auto match_contract_mul = [&](const Def* candidate,
                                  const Def* add_mode) -> std::optional<std::pair<const Def*, const Def*>> {
        auto mul = Axm::isa<math::arith>(candidate);
        if (!mul || mul.id() != math::arith::mul) return std::nullopt;
        auto [mul_mode, args] = mul->uncurry_args<2>();
        if (!has_contract(mul_mode) || Lit::as(mul_mode) != Lit::as(add_mode)) return std::nullopt;
        auto [a, b] = args->projs<2>();
        return std::pair{a, b};
    };

    if (auto load = Axm::isa<mem::load>(def); load && simd_ptr_type(load->arg(1)->type())) {
        emit_unsafe(load->arg(0));
        auto ptr_t = *simd_ptr_type(load->arg(1)->type());
        auto width = *simd_f32_arr(Axm::as<mem::Ptr>(load->arg(1)->type())->arg(0));
        return bb.assign(name, "load <{} x float>, {} {}", width, ptr_t, emit(load->arg(1)));
    } else if (auto store = Axm::isa<mem::store>(def); store && simd_ptr_type(store->arg(1)->type())) {
        emit_unsafe(store->arg(0));
        auto ptr_t = *simd_ptr_type(store->arg(1)->type());
        auto width = *simd_f32_arr(Axm::as<mem::Ptr>(store->arg(1)->type())->arg(0));
        std::print(bb.body().emplace_back(), "store <{} x float> {}, {} {}", width, emit(store->arg(2)), ptr_t,
                   emit(store->arg(1)));
        return {};
    } else if (auto arith = Axm::isa<math::arith>(def);
        arith && arith.id() == math::arith::add && math::isa_f(arith->type())) {
        auto [mode, args] = arith->uncurry_args<2>();
        if (has_contract(mode)) {
            auto [a, b] = args->projs<2>();
            const Def* mul_lhs = nullptr;
            const Def* mul_rhs = nullptr;
            const Def* addend  = nullptr;

            if (auto mul = match_contract_mul(a, mode)) {
                mul_lhs = mul->first;
                mul_rhs = mul->second;
                addend  = b;
            } else if (auto mul = match_contract_mul(b, mode)) {
                mul_lhs = mul->first;
                mul_rhs = mul->second;
                addend  = a;
            }

            if (mul_lhs) {
                auto t     = convert(arith->type());
                auto width = *math::isa_f(arith->type());
                declare("{} @llvm.fma.f{}({}, {}, {})", t, width, t, t, t);
                return bb.assign(name, "call {} @llvm.fma.f{}({} {}, {} {}, {} {})", t, width, t, emit(mul_lhs), t,
                                 emit(mul_rhs), t, emit(addend));
            }
        }
    } else if (auto mslot = Axm::isa<mem::mslot>(def)) {
        auto [T, a] = mslot->decurry()->args<2>();
        if (Lit::as(a) == shared_as) {
            name = "@" + def->unique_name();
            emit_unsafe(mslot->arg(0));
            std::print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", name, a, convert(T));
            return name;
        }
        if (Lit::as(a) == local_as) {
            emit_unsafe(mslot->arg(0));
            auto type = convert(T, false);
            std::print(bb.body().emplace_back(), "{} = alloca {}", name, type);
            return name;
        }
    } else if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        declare("void @llvm.nvvm.barrier0()");

        emit_unsafe(sync_work_items->arg(0));
        emit_unsafe(sync_work_items->arg(1));
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.barrier0()");
        return name;
    } else if (auto cp_async = Axm::isa<gpu::cp_async>(def)) {
        auto [T] = cp_async->decurry()->args<1>();
        auto size = Lit::as(world().call(core::trait::size, T));
        if (size != 4 && size != 8 && size != 16)
            error("gpu.cp_async supports only 4, 8, or 16 byte copies, got {} bytes for {}", size, T);
        emit_unsafe(cp_async->arg(0));
        emit_unsafe(cp_async->arg(1));
        auto dst = emit(cp_async->arg(2));
        auto src = emit(cp_async->arg(3));
        auto intrinsic = std::format("llvm.nvvm.cp.async.ca.shared.global.{}", size);
        declare("void @{}(ptr addrspace(3), ptr addrspace(1))", intrinsic);
        std::print(bb.body().emplace_back(), "call void @{}(ptr addrspace(3) {}, ptr addrspace(1) {})", intrinsic, dst, src);
        return name;
    } else if (auto commit = Axm::isa<gpu::cp_async_commit_group>(def)) {
        declare("void @llvm.nvvm.cp.async.commit.group()");
        emit_unsafe(commit->arg(0));
        emit_unsafe(commit->arg(1));
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.commit.group()");
        return name;
    } else if (auto wait = Axm::isa<gpu::cp_async_wait_group>(def)) {
        auto [groups] = wait->decurry()->args<1>();
        if (!Lit::isa(groups)) error("gpu.cp_async_wait_group requires a literal group count, got {}", groups);
        auto count = Lit::as(groups);
        declare("void @llvm.nvvm.cp.async.wait.group(i32 immarg)");
        emit_unsafe(wait->arg(0));
        emit_unsafe(wait->arg(1));
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.wait.group(i32 {})", count);
        return name;
    }
    return std::nullopt;
}

void emit_host(World& world, std::ostream& ostream, std::optional<std::string> device_fatbin_file) {
    HostEmitter emitter(world, ostream, device_fatbin_file);
    emitter.run();
}

DeviceEmitFlags emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();

    return DeviceEmitFlags{
        .uses_libdevice = emitter.is_using_libdevice(),
    };
}

} // namespace mim::plug::ll_nvptx
