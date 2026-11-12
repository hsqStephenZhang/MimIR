#include "mim/plug/nvptx/phase/nvptx.h"

#include <algorithm>
#include <format>
#include <map>

#include <fe/log.h>
#include <fe/sys.h>

#include <mim/driver.h>
#include <mim/lam.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/nvptx/nvptx.h"

using namespace std::string_literals;

namespace mim::plug::nvptx {

namespace core = mim::plug::core;
namespace ll   = mim::plug::ll;
namespace mem  = mim::plug::mem;
namespace gpu  = mim::plug::gpu;

class HostEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    HostEmitter(World& world,
                std::ostream& ostream,
                std::optional<std::string> device_fatbin_file,
                std::map<std::string, u64> dynamic_shared)
        : Super(world, "llvm_nvptx_host_emitter", ostream)
        , device_fatbin_file_(device_fatbin_file)
        , dynamic_shared_(std::move(dynamic_shared)) {}

    void start() final;
    void find_kernels(const Def*);

    std::string prepare() override;
    void emit_epilogue(Lam*) override;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

protected:
    std::string convert(const Def*, bool simd = true) override;

private:
    static constexpr std::string_view mod_name_          = "@.mimir_cu_mod";
    static constexpr std::string_view ctx_name_          = "@.mimir_cu_ctx";
    static constexpr std::string_view fatbin_name_       = "@.fatbin";
    static constexpr std::string_view kernel_array_name_ = "@.mimir_kernels";
    static constexpr std::string_view kernel_name_prefix = "@.kname.";

    void emit_cu_error_handling(ll::BB&, const std::string&);
    void emit_gpu_setup(ll::BB&, const std::string& name);
    void emit_gpu_teardown(ll::BB&, const std::string& name);

    std::optional<std::string> device_fatbin_file_;
    std::map<std::string, u64> dynamic_shared_;
    LamMap<int> kernel_ids_;
    bool cu_globals_declared_ = false;

    DefSet analyzed_;
    /// Externals wrapped in their own GPU setup/teardown - see HostEmitter::start().
    LamSet gpu_externals_;
};

class DeviceEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream, bool async_copy)
        : Super(world, "llvm_nvptx_device_emitter", ostream)
        , async_copy_(async_copy) {
        simd_ = false;
    }

    bool use_simd() const override { return false; }

    void start() final;

    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;
    void emit_epilogue(Lam*) override;

    bool is_using_libdevice() const { return uses_libdevice; }
    const std::map<std::string, u64>& dynamic_shared() const { return dynamic_shared_; }
    const std::string& get_extra_flags() const { return extra_flags; }

private:
    /// Shared and constant slots live in a module-scope global in their address space; generic ones stay per thread.
    std::string emit_slot(ll::BB& bb, const App* app, const Def* pointee, const Def* addr_space) override {
        if (Lit::as(addr_space) == 0) return ll::Emitter::emit_slot(bb, app, pointee, addr_space);
        auto v_ptr = "@" + app->unique_name() + ".slot";
        std::println(vars_decls_, "{} = internal addrspace({}) global {} undef, align 16", v_ptr, addr_space,
                     convert(pointee, false));
        return v_ptr;
    }

    LamSet kernels_;
    /// Under `-X nvptx:profile=1` a kernel sums the cycles before and at each barrier site; the first thread prints
    /// them.
    bool profile_                      = false;
    u64 profile_sites_                 = 0;
    static constexpr u64 Profile_Slots = 64;
    /// A kernel whose shared slots exceed the static limit, or overlay panels, carves them out of one array, at these
    /// offsets.
    DefMap<u64> dynamic_slots_;
    std::string dynamic_array_;
    std::map<std::string, u64> dynamic_shared_;
    /// `cp.async` needs sm_80; below it a `gpu.store_async` is a plain store.
    bool async_copy_;

    bool uses_libdevice = false;
    std::string extra_flags;
};

namespace {
// The shared memory a kernel may declare statically.
constexpr u64 Static_Shared_Limit = 48 * 1024;

// Kernel arguments are typed global pointers; offsets and casts of them stay in global memory.
bool is_global(const Def* ptr) {
    while (true) {
        if (auto type = Axm::isa<mem::Ptr>(ptr->type()); type && Lit::isa(type->arg(2, 1)) == 1) return true;
        if (auto lea = Axm::isa<mem::lea>(ptr))
            ptr = lea->arg(2, 0);
        else if (auto cast = Axm::isa<core::bitcast>(ptr))
            ptr = cast->arg();
        else if (auto aligned = Axm::isa<mem::align>(ptr))
            ptr = aligned->arg(2, 0);
        else
            return false;
    }
}

// The global pointer @p val is loaded from.
// A lane of a loaded vector stays a plain store: copying lanes apart takes a request per lane.
const Def* global_source(const Def* val) {
    auto read = val->isa<Extract>();
    if (!read || Lit::isa(read->index()) != 1) return nullptr;
    auto load = Axm::isa<mem::load>(read->tuple());
    if (!load || !is_global(load->arg(2, 1))) return nullptr;
    return load->arg(2, 1);
}

bool reaches_if(const Def* def, DefSet& seen, std::predicate<const Def*> auto pred) {
    if (auto [_, ins] = seen.emplace(def); !ins) return false;
    if (pred(def)) return true;
    for (auto d : def->deps())
        if (reaches_if(d, seen, pred)) return true;
    return false;
}

bool is_gpu_auto_init(const Def* def) { return Axm::isa<gpu::auto_init>(def) != nullptr; }
} // namespace

void HostEmitter::start() {
    for (auto def : world().externals().muts())
        find_kernels(def);

    for (auto [kernel, kid] : kernel_ids_) {
        auto name = id(kernel).substr(1);
        std::println(vars_decls_, "{}{} = private constant [{} x i8] c\"{}\\00\"", kernel_name_prefix, kid,
                     name.size() + 1, name);
    }
    std::println(vars_decls_, "{} = dso_local global [{} x ptr] zeroinitializer", kernel_array_name_,
                 kernel_ids_.size());

    LamSet gpu_touching;
    for (auto mut : world().externals().muts()) {
        auto lam = mut->isa_mut<Lam>();
        if (!lam || !lam->ret_pi()) continue;
        DefSet seen;
        if (reaches_if(lam, seen, is_gpu_auto_init)) gpu_touching.emplace(lam);
    }
    // Only wrap the ones not themselves called by another GPU-touching external: a caller's own
    // setup/teardown already covers whatever GPU-touching external(s) it calls.
    for (auto lam : gpu_touching) {
        auto called_by_other = std::ranges::any_of(gpu_touching, [&](auto other) {
            if (other == lam) return false;
            DefSet seen;
            return reaches_if(other, seen, [lam](const Def* d) { return d == lam; });
        });
        if (!called_by_other) gpu_externals_.emplace(lam);
    }

    Super::start();
}

void HostEmitter::find_kernels(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    for (auto d : def->deps())
        find_kernels(d);

    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel     = launch->decurry()->decurry()->arg();
        auto kernel_lam = kernel->expect_mut<Lam>("the kernel passed to `gpu.launch` to be a mutable lambda");
        if (kernel_ids_.contains(kernel_lam)) return;
        auto kid                = kernel_ids_.size();
        kernel_ids_[kernel_lam] = kid;
    }
}

constexpr auto Cu_Init                = "cuInit";
constexpr auto Cu_Ctx_Create          = "cuCtxCreate_v4";
constexpr auto Cu_Ctx_Destroy         = "cuCtxDestroy_v2";
constexpr auto Cu_Device_Get          = "cuDeviceGet";
constexpr auto Cu_Launch_Kernel       = "cuLaunchKernel_ptsz";
constexpr auto Cu_Mem_Alloc           = "cuMemAlloc_v2";
constexpr auto Cu_Mem_Alloc_Async     = "cuMemAllocAsync_ptsz";
constexpr auto Cu_Mem_Free            = "cuMemFree_v2";
constexpr auto Cu_Mem_Free_Async      = "cuMemFreeAsync_ptsz";
constexpr auto Cu_Memcpy_Htod         = "cuMemcpyHtoD_v2";
constexpr auto Cu_Memcpy_Htod_Async   = "cuMemcpyHtoDAsync_v2_ptsz";
constexpr auto Cu_Memcpy_Dtoh         = "cuMemcpyDtoH_v2";
constexpr auto Cu_Memcpy_Dtoh_Async   = "cuMemcpyDtoHAsync_v2_ptsz";
constexpr auto Cu_Module_Load_Fatbin  = "cuModuleLoadFatBinary";
constexpr auto Cu_Module_Get_Function = "cuModuleGetFunction";
constexpr auto Cu_Func_Set_Attribute  = "cuFuncSetAttribute";
constexpr auto Cu_Max_Dynamic_Shared  = 8; // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
constexpr auto Cu_Shared_Carveout     = 9; // CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
constexpr auto Cu_Module_Unload       = "cuModuleUnload";
constexpr auto Cu_Stream_Create       = "cuStreamCreate";
constexpr auto Cu_Stream_Destroy      = "cuStreamDestroy_v2";
constexpr auto Cu_Stream_Sync         = "cuStreamSynchronize_ptsz";

void HostEmitter::emit_cu_error_handling(ll::BB& bb, const std::string& cu_result) {
    // Offload the CUresult check to the C runtime wrapper `mim_cu_check` (see rt/mim_cuda_rt.c)
    // instead of open-coding it here; showcases the C-runtime system on the nvptx backend.
    declare_rt("void @mim_cu_check(i32)");
    std::print(bb.body().emplace_back(), "call void @mim_cu_check(i32 {})", cu_result);
}

void HostEmitter::emit_gpu_setup(ll::BB& bb, const std::string& name) {
    auto dev_num   = 0; // TODO: consider parameterizing this
    auto ctx_flags = 0; // TODO: consider parameterizing this

    declare("i32 @{}(i32)", Cu_Init);
    auto init_res = bb.assign(name + "_init_res", "call i32 @{}(i32 0)", Cu_Init);
    emit_cu_error_handling(bb, init_res);

    declare("i32 @{}(ptr, i32)", Cu_Device_Get);
    auto dev_ptr     = bb.assign(name + "_dev_ptr", "alloca i32");
    auto dev_get_res = bb.assign(name + "_get_res", "call i32 @{}(ptr {}, i32 {})", Cu_Device_Get, dev_ptr, dev_num);
    emit_cu_error_handling(bb, dev_get_res);

    declare("i32 @{}(ptr, ptr, i32, i32)", Cu_Ctx_Create);
    if (!cu_globals_declared_) std::println(vars_decls_, "{} = global ptr null", ctx_name_);
    auto dev     = bb.assign(name + "_dev", "load i32, ptr {}", dev_ptr);
    auto ctx_res = bb.assign(name + "_ctx_res", "call i32 @{}(ptr {}, ptr null, i32 {}, i32 {})", Cu_Ctx_Create,
                             ctx_name_, ctx_flags, dev);
    emit_cu_error_handling(bb, ctx_res);

    declare("i32 @{}(ptr, ptr)", Cu_Module_Load_Fatbin);
    if (!cu_globals_declared_) {
        std::println(vars_decls_, "{} = global ptr null", mod_name_);
        if (device_fatbin_file_.has_value()) {
            std::ifstream fatbin_file(device_fatbin_file_.value(), std::ios::binary);
            if (!fatbin_file)
                fe::throwf(MIM_NVPTX_BE "could not open `{}` as binary file", device_fatbin_file_.value());

            auto start = std::istreambuf_iterator<char>(fatbin_file);
            auto end   = std::istreambuf_iterator<char>();
            std::vector<u8> fatbin_bytes(start, end);

            std::print(vars_decls_, "{} = private constant [{} x i8] c\"", fatbin_name_, fatbin_bytes.size());
            for (auto byte : fatbin_bytes) {
                bool invalid_cstr_char = byte == '"' || byte == '\\';
                if (std::isprint(byte) && !invalid_cstr_char)
                    std::print(vars_decls_, "{:c}", byte);
                else
                    std::print(vars_decls_, "\\{:02x}", byte);
            }
            std::println(vars_decls_, "\"");
        } else {
            std::println(vars_decls_, "; Add the bytes of your compiled nvptx fatbin binary here:");
            std::println(vars_decls_,
                         "{} = private constant [YOUR_FATBIN_DATA_SIZE_GOES_HERE x i8] YOUR_FATBIN_DATA_GOES_HERE",
                         fatbin_name_);
        }
        cu_globals_declared_ = true;
    }
    auto mod_res
        = bb.assign(name + "_mod_res", "call i32 @{}(ptr {}, ptr {})", Cu_Module_Load_Fatbin, mod_name_, fatbin_name_);
    emit_cu_error_handling(bb, mod_res);
    auto mod_inner = bb.assign(name + "_mod_inner", "load ptr, ptr {}", mod_name_);

    declare("i32 @{}(ptr, ptr, ptr)", Cu_Module_Get_Function);
    for (auto [kernel, kid] : kernel_ids_) {
        auto kname    = id(kernel).substr(1);
        auto func_ptr = bb.assign(name + "_" + kname + "_funcptr", "getelementptr inbounds ptr, ptr {}, i64 {}",
                                  kernel_array_name_, kid);
        auto func_res = bb.assign(name + "_" + kname + "_getfuncres", "call i32 @{}(ptr {}, ptr {}, ptr {}{})",
                                  Cu_Module_Get_Function, func_ptr, mod_inner, kernel_name_prefix, kid);
        emit_cu_error_handling(bb, func_res);
        if (auto bytes = dynamic_shared_.find(kname); bytes != dynamic_shared_.end()) {
            declare("i32 @{}(ptr, i32, i32)", Cu_Func_Set_Attribute);
            auto func = bb.assign(name + "_" + kname + "_func", "load ptr, ptr {}", func_ptr);
            auto res  = bb.assign(name + "_" + kname + "_setattr", "call i32 @{}(ptr {}, i32 {}, i32 {})",
                                  Cu_Func_Set_Attribute, func, Cu_Max_Dynamic_Shared, bytes->second);
            emit_cu_error_handling(bb, res);
            // Without a preference the driver may split the SM's storage toward L1, which caps the resident groups.
            auto carveout = bb.assign(name + "_" + kname + "_carveout", "call i32 @{}(ptr {}, i32 {}, i32 100)",
                                      Cu_Func_Set_Attribute, func, Cu_Shared_Carveout);
            emit_cu_error_handling(bb, carveout);
        }
    }
}

void HostEmitter::emit_gpu_teardown(ll::BB& bb, const std::string& name) {
    declare("i32 @{}(ptr)", Cu_Module_Unload);
    std::print(bb.body().emplace_back(), "{}_mod = load ptr, ptr {}", name, mod_name_);
    std::print(bb.body().emplace_back(), "{}_mod_unload_res = call i32 @{}(ptr {}_mod)", name, Cu_Module_Unload, name);
    emit_cu_error_handling(bb, name + "_mod_unload_res");

    declare("i32 @{}(ptr)", Cu_Ctx_Destroy);
    std::print(bb.body().emplace_back(), "{}_ctx = load ptr, ptr {}", name, ctx_name_);
    std::print(bb.body().emplace_back(), "{}_ctx_destroy_res = call i32 @{}(ptr {}_ctx)", name, Cu_Ctx_Destroy, name);
    emit_cu_error_handling(bb, name + "_ctx_destroy_res");
}

std::string HostEmitter::prepare() {
    auto name = Super::prepare();
    // Append to root()'s own BB, not func_impls_: Emitter::finalize_impl writes it out only once.
    if (gpu_externals_.contains(root())) emit_gpu_setup(lam2bb_[root()], "%" + root()->unique_name());
    return name;
}

void HostEmitter::emit_epilogue(Lam* lam) {
    // Must run first to force emission of the return value's own dependencies (e.g. gpu.free) into bb.body().
    Super::emit_epilogue(lam);
    if (gpu_externals_.contains(root())) {
        // lam, not root(): a function can have several return blocks, and LLVM names are function-scoped.
        if (auto app = lam->body()->isa<App>(); app && app->callee() == root()->ret_var())
            emit_gpu_teardown(lam2bb_[lam], "%" + lam->unique_name());
    }
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

std::optional<std::string> HostEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    if (auto default_stream = Axm::isa<gpu::default_stream>(def)) {
        return "null";
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        auto mem_val = emit_unsafe(init->arg());
        emit_gpu_setup(bb, name);
        return mem_val;
    } else if (auto deinit = Axm::isa<gpu::deinit>(def)) {
        emit_unsafe(deinit->arg(3, 0));
        emit_unsafe(deinit->arg(3, 1));
        emit_gpu_teardown(bb, name);
        return ""s;
    } else if (auto auto_init = Axm::isa<gpu::auto_init>(def)) {
        return emit_unsafe(auto_init->arg());
    } else if (auto auto_deinit = Axm::isa<gpu::auto_deinit>(def)) {
        // emit_unsafe on `global` keeps any gpu.free threaded through it reachable.
        emit_unsafe(auto_deinit->arg(3, 1));
        emit_unsafe(auto_deinit->arg(3, 2));
        return emit_unsafe(auto_deinit->arg(3, 0));
    } else if (auto stream_init = Axm::isa<gpu::stream_init>(def)) {
        declare("i32 @{}(ptr, i32)", Cu_Stream_Create);

        emit_unsafe(stream_init->arg(3, 0));
        emit_unsafe(stream_init->arg(3, 1));
        auto stream_ptr = emit(stream_init->arg(3, 2));

        auto res = bb.assign(name, "call i32 @{}(ptr {}, i32 0)", Cu_Stream_Create, stream_ptr);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        declare("i32 @{}(ptr)", Cu_Stream_Destroy);

        emit_unsafe(stream_deinit->arg(3, 0));
        emit_unsafe(stream_deinit->arg(3, 1));
        auto stream = emit(stream_deinit->arg(3, 2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", Cu_Stream_Destroy, stream);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare("i32 @{}(ptr)", Cu_Stream_Sync);

        emit_unsafe(stream_sync->arg(3, 0));
        emit_unsafe(stream_sync->arg(3, 1));
        auto stream = emit(stream_sync->arg(3, 2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", Cu_Stream_Sync, stream);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        bool is_async;
        switch (alloc.id()) {
            case gpu::alloc::block: is_async = false; break;
            case gpu::alloc::asyn: is_async = true; break;
            default: fe::throwf(MIM_NVPTX_BE "unhandled `gpu.alloc` id in `{}`", def);
        }

        // `.block` and `.asyn` differ in arity, so the count has to follow the variant.
        auto n = is_async ? 2_n : 1_n;

        if (is_async)
            declare("i32 @{}(ptr, i64, ptr)", Cu_Mem_Alloc_Async);
        else
            declare("i32 @{}(ptr, i64)", Cu_Mem_Alloc);

        emit_unsafe(alloc->arg(n, 0));
        auto alloc_t    = alloc->decurry()->arg();
        World& w        = alloc_t->world();
        auto type_size  = w.call(core::trait::size, alloc_t);
        auto alloc_size = emit(type_size);

        auto ptr_t = convert(Axm::expect<mem::Ptr>(def->proj(2, 1)->type(), "a `mem.Ptr`"));

        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        std::string alloc_res;
        if (is_async) {
            auto stream = emit(alloc->arg(n, 1));
            alloc_res   = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {}, ptr {})", Cu_Mem_Alloc_Async, alloc_ptr,
                                    alloc_size, stream);
        } else
            alloc_res = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {})", Cu_Mem_Alloc, alloc_ptr, alloc_size);

        emit_cu_error_handling(bb, alloc_res);
        return bb.assign(name, "load {}, {} addrspace(0)* {}", ptr_t, ptr_t, alloc_ptr);
    } else if (auto free = Axm::isa<gpu::free>(def)) {
        bool is_async;
        switch (free.id()) {
            case gpu::free::block: is_async = false; break;
            case gpu::free::asyn: is_async = true; break;
            default: fe::throwf(MIM_NVPTX_BE "unhandled `gpu.free` id in `{}`", def);
        }

        // `.block` and `.asyn` differ in arity, so the count has to follow the variant.
        auto n = is_async ? 3_n : 2_n;

        if (is_async)
            declare("i32 @{}(i64, ptr)", Cu_Mem_Free_Async);
        else
            declare("i32 @{}(i64)", Cu_Mem_Free);

        emit_unsafe(free->arg(n, 0));
        auto ptr = emit(free->arg(n, 1));

        std::string free_res;
        if (is_async) {
            auto stream = emit(free->arg(n, 2));
            free_res    = bb.assign(name + "res", "call i32 @{}(i64 {}, ptr {})", Cu_Mem_Free_Async, ptr, stream);
        } else
            free_res = bb.assign(name + "res", "call i32 @{}(i64 {})", Cu_Mem_Free, ptr);

        emit_cu_error_handling(bb, free_res);
        return free_res;
    } else if (auto copy_to_device = Axm::isa<gpu::copy_to_device>(def)) {
        bool is_async;
        switch (copy_to_device.id()) {
            case gpu::copy_to_device::block: is_async = false; break;
            case gpu::copy_to_device::asyn: is_async = true; break;
            default: fe::throwf(MIM_NVPTX_BE "unhandled `gpu.copy_to_device` id in `{}`", def);
        }

        // `.block` and `.asyn` differ in arity, so the count has to follow the variant.
        auto n = is_async ? 5_n : 4_n;

        if (is_async)
            declare("i32 @{}(i64, ptr, i64, ptr)", Cu_Memcpy_Htod_Async);
        else
            declare("i32 @{}(i64, ptr, i64)", Cu_Memcpy_Htod);

        auto type      = copy_to_device->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_device->arg(n, 0));
        emit_unsafe(copy_to_device->arg(n, 1));
        auto host_ptr = emit(copy_to_device->arg(n, 2));
        auto dev_ptr  = emit(copy_to_device->arg(n, 3));
        auto size     = emit(type_size);

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_device->arg(n, 4));
            copy_res    = bb.assign(name + "res", "call i32 @{}(i64 {}, ptr {}, i64 {}, ptr {})", Cu_Memcpy_Htod_Async,
                                    dev_ptr, host_ptr, size, stream);
        } else
            copy_res = bb.assign(name + "res", "call i32 @{}(i64 {}, ptr {}, i64 {})", Cu_Memcpy_Htod, dev_ptr,
                                 host_ptr, size);

        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto copy_to_host = Axm::isa<gpu::copy_to_host>(def)) {
        bool is_async;
        switch (copy_to_host.id()) {
            case gpu::copy_to_host::block: is_async = false; break;
            case gpu::copy_to_host::asyn: is_async = true; break;
            default: fe::throwf(MIM_NVPTX_BE "unhandled `gpu.copy_to_host` id in `{}`", def);
        }

        // `.block` and `.asyn` differ in arity, so the count has to follow the variant.
        auto n = is_async ? 5_n : 4_n;
        if (is_async)
            declare("i32 @{}(ptr, i64, i64, ptr)", Cu_Memcpy_Dtoh_Async);
        else
            declare("i32 @{}(ptr, i64, i64)", Cu_Memcpy_Dtoh);

        auto type      = copy_to_host->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_host->arg(n, 0));
        emit_unsafe(copy_to_host->arg(n, 1));
        auto dev_ptr  = emit(copy_to_host->arg(n, 2));
        auto host_ptr = emit(copy_to_host->arg(n, 3));
        auto size     = emit(type_size);

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_host->arg(n, 4));
            copy_res    = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {}, i64 {}, ptr {})", Cu_Memcpy_Dtoh_Async,
                                    host_ptr, dev_ptr, size, stream);
        } else
            copy_res = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {}, i64 {})", Cu_Memcpy_Dtoh, host_ptr,
                                 dev_ptr, size);

        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        // TODO: rewrite to use modern cuLaunchKernelEx instead
        declare("i32 @{}(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)", Cu_Launch_Kernel);

        auto [implicits, launch_config, kernel_def, arg_def, func_args] = launch->uncurry_args<5>();
        auto [n_groups_def, n_items_def, stream_def, m, MT]             = launch_config->projs<5>();
        auto [mem, ret_lam_def]                                         = func_args->projs<2>();

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) fe::throwf(MIM_NVPTX_BE "kernel `{}` is not a lambda", kernel_def);
        if (!kernel_ids_.contains(lam)) fe::throwf(MIM_NVPTX_BE "unknown kernel `{}`", lam);
        auto kid = kernel_ids_[lam];

        u64 shared_mem_bytes = 0;
        if (auto smem_count = Lit::expect(m, "a shared-memory allocation count")) {
            if (smem_count != 1)
                fe::throwf(MIM_NVPTX_BE "only one dynamic shared-memory allocation is allowed per kernel");
            shared_mem_bytes = Lit::expect(world().call(core::trait::size, MT), "a shared-memory size");
        }

        if (auto bytes = dynamic_shared_.find(id(lam).substr(1)); bytes != dynamic_shared_.end()) {
            if (shared_mem_bytes) fe::throwf(MIM_NVPTX_BE "kernel `{}` has dynamic shared memory twice", lam);
            shared_mem_bytes = bytes->second;
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
                                  kernel_ids_.size(), kernel_ids_.size(), kernel_array_name_, kid);
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
                        Cu_Launch_Kernel, func_inner, n_groups, n_items, shared_mem_bytes, stream, args_inner);
        emit_cu_error_handling(bb, launch_res);
        return ret_lam;
    }
    return std::nullopt;
}

void DeviceEmitter::start() {
    profile_ = world().driver().arg_nat("nvptx", "profile").value_or(0) != 0;
    for (auto kernel : world().externals().muts()) {
        auto kernel_lam = kernel->expect_mut<Lam>("an external kernel to be a mutable lambda");
        kernels_.emplace(kernel_lam);
    }
    Super::start();
}

std::string DeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    std::print(func_impls_, "define ptx_kernel {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    auto [m1, m3, m4, m5, group_id, item_id, smem, arg, ret_lam] = kernel->vars<9>();

    dynamic_slots_.clear();
    auto slots = DefVec();
    auto seen  = DefSet();
    for (auto todo = DefVec{kernel}; !todo.empty();) {
        auto def = todo.back();
        todo.pop_back();
        if (!seen.emplace(def).second) continue;
        if (Axm::isa<gpu::slot_shared>(def)) slots.push_back(def);
        for (auto dep : def->deps())
            todo.push_back(dep);
    }
    auto bytes = [](const Def* slot) {
        auto size = Lit::as(core::op(core::trait::size, slot->as<App>()->decurry()->arg()));
        return (size + 15) / 16 * 16;
    };
    u64 plain = 0;
    for (auto slot : slots)
        plain += bytes(slot);
    // The panels of different constructs are never live together, so each construct's panels may start at one base.
    // Overlaying hides that the panels never alias, so it only pays where the slots outgrow the static limit.
    u64 total   = 0;
    auto panels = std::map<u64, DefVec>();
    for (auto slot : slots) {
        if (auto scope = Axm::isa<gpu::panel_scope>(slot->as<App>()->arg()); scope && plain > Static_Shared_Limit) {
            panels[Lit::as(scope->decurry()->arg())].push_back(slot);
            continue;
        }
        dynamic_slots_[slot] = total;
        total += bytes(slot);
    }
    u64 arena = 0;
    for (auto& [_, group] : panels) {
        u64 offset = total;
        for (auto slot : group) {
            dynamic_slots_[slot] = offset;
            offset += bytes(slot);
        }
        arena = std::max(arena, offset - total);
    }
    total += arena;
    if (plain > Static_Shared_Limit) {
        auto kname             = id(kernel).substr(1);
        dynamic_array_         = "@" + kname + "_dynamic_shared";
        dynamic_shared_[kname] = total;
        std::println(vars_decls_, "{} = external addrspace(3) global [0 x i8], align 16", dynamic_array_);
    } else {
        dynamic_slots_.clear();
    }

    auto arg_name = id(arg);
    locals_[arg]  = arg_name;
    std::println(func_impls_, "{} {}) {{", convert(arg->type()), arg_name);

    auto& bb = lam2bb_[kernel];

    auto register_sreg_idx = [&](const Def* def, std::string_view sreg) {
        auto name        = id(def);
        auto type        = def->type();
        auto type_name   = convert(type);
        auto opt_idx_lit = Idx::isa_lit(type);
        if (!opt_idx_lit)
            fe::throwf(MIM_NVPTX_BE "type of `{}` must be a statically-sized `Idx` but is `{}`", def, type);
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
            fe::throwf(MIM_NVPTX_BE "warp ID too large; must fit into `I32`");
        }
    };
    register_sreg_idx(group_id, "ctaid.x");
    register_sreg_idx(item_id, "tid.x");
    if (profile_) {
        declare("i64 @llvm.nvvm.read.ptx.sreg.clock64()");
        declare("void @llvm.memset.p0.i64(ptr, i8, i64, i1)");
        profile_sites_ = 0;
        bb.assign("%prof.acc", "alloca [{} x i64]", 2 * Profile_Slots);
        std::print(bb.body().emplace_back(), "call void @llvm.memset.p0.i64(ptr %prof.acc, i8 0, i64 {}, i1 false)",
                   16 * Profile_Slots);
        bb.assign("%prof.last", "alloca i64");
        bb.assign("%prof.t0", "call i64 asm sideeffect \"mov.u64 $0, %clock64;\", \"=l\"()");
        std::print(bb.body().emplace_back(), "store i64 %prof.t0, ptr %prof.last");
    }

    auto shared_as = Lit::expect(world().annex<gpu::addr_space_shared>(), "the shared address space");
    if (auto sigma = smem->type()->isa<Sigma>()) {
        if (sigma->num_ops() != 0)
            fe::throwf(MIM_NVPTX_BE "shared-memory variable must be an empty sigma, but got `{}`", smem->type());
    } else {
        auto ptr    = Axm::expect<mem::Ptr>(smem->type(), "a shared-memory pointer type");
        auto [T, a] = ptr->args<2>();
        if (Lit::expect(a, "an address space") != shared_as)
            fe::throwf(MIM_NVPTX_BE "shared-memory variable must live in the shared address space, but got `{}`",
                       smem->type());
        auto name     = "@" + smem->unique_name();
        locals_[smem] = name;
        std::println(vars_decls_, "{} = internal addrspace({}) global {} undef, align 16", name, a, convert(T, false));
    }

    return kernel->unique_name();
}

// The first thread of the first work-group prints each barrier site's cycles before (`c`) and at (`w`) the barrier.
void DeviceEmitter::emit_epilogue(Lam* lam) {
    auto app = lam->body()->isa<App>();
    if (profile_ && kernels_.contains(root()) && app && app->callee() == root()->ret_var()) {
        auto& bb    = lam2bb_[lam];
        auto label  = id(lam).substr(1) + ".prof";
        auto format = "@" + root()->unique_name() + ".prof.format";
        std::println(
            vars_decls_,
            "{} = private unnamed_addr addrspace(1) constant [20 x i8] c\"prof %2d %c %12lld\\0A\\00\", align 1",
            format);
        declare("i32 @vprintf(ptr, ptr)");
        declare("i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
        declare("i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
        auto tid   = bb.assign("%" + label + ".tid", "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
        auto cta   = bb.assign("%" + label + ".cta", "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
        auto who   = bb.assign("%" + label + ".who", "or i32 {}, {}", tid, cta);
        auto first = bb.assign("%" + label + ".first", "icmp eq i32 {}, 0", who);
        auto args  = bb.assign("%" + label + ".args", "alloca {{ i32, i32, i64 }}");
        std::print(bb.body().emplace_back(), "br i1 {}, label %{}.print, label %{}.done", first, label, label);
        std::print(bb.body().emplace_back(), "{}.print:", label);
        for (u64 slot = 0; slot != 2 * profile_sites_ && slot != 2 * Profile_Slots; ++slot) {
            auto n  = "%" + label + "." + std::to_string(slot);
            auto at = bb.assign(n + ".at", "getelementptr [{} x i64], ptr %prof.acc, i64 0, i64 {}", 2 * Profile_Slots,
                                slot);
            auto v  = bb.assign(n + ".v", "load i64, ptr {}", at);
            std::print(bb.body().emplace_back(), "store i32 {}, ptr {}", slot / 2, args);
            auto k = bb.assign(n + ".k", "getelementptr {{ i32, i32, i64 }}, ptr {}, i32 0, i32 1", args);
            std::print(bb.body().emplace_back(), "store i32 {}, ptr {}", int(slot % 2 ? 'w' : 'c'), k);
            auto x = bb.assign(n + ".x", "getelementptr {{ i32, i32, i64 }}, ptr {}, i32 0, i32 2", args);
            std::print(bb.body().emplace_back(), "store i64 {}, ptr {}", v, x);
            bb.assign(n + ".r", "call i32 @vprintf(ptr addrspacecast (ptr addrspace(1) {} to ptr), ptr {})", format,
                      args);
        }
        std::print(bb.body().emplace_back(), "br label %{}.done", label);
        std::print(bb.body().emplace_back(), "{}.done:", label);
    }
    Super::emit_epilogue(lam);
}

std::optional<std::string> DeviceEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        declare("void @llvm.nvvm.barrier0()");

        emit_unsafe(sync_work_items->arg(2, 0));
        emit_unsafe(sync_work_items->arg(2, 1));
        auto site    = profile_ && kernels_.contains(root()) ? profile_sites_++ : Profile_Slots;
        auto measure = [&](std::string tag, u64 slot) {
            auto now = bb.assign(name + ".prof." + tag, "call i64 asm sideeffect \"mov.u64 $0, %clock64;\", \"=l\"()");
            auto at  = bb.assign(name + ".prof." + tag + "at", "getelementptr [{} x i64], ptr %prof.acc, i64 0, i64 {}",
                                 2 * Profile_Slots, slot);
            auto was = bb.assign(name + ".prof." + tag + "was", "load i64, ptr {}", at);
            auto last = bb.assign(name + ".prof." + tag + "last", "load i64, ptr %prof.last");
            auto d    = bb.assign(name + ".prof." + tag + "d", "sub i64 {}, {}", now, last);
            auto sum  = bb.assign(name + ".prof." + tag + "sum", "add i64 {}, {}", was, d);
            std::print(bb.body().emplace_back(), "store i64 {}, ptr {}", sum, at);
            std::print(bb.body().emplace_back(), "store i64 {}, ptr %prof.last", now);
        };
        if (site < Profile_Slots) measure("c", 2 * site);
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.barrier0()");
        if (site < Profile_Slots) measure("w", 2 * site + 1);
        return name;
    } else if (auto store = Axm::isa<gpu::store_async>(def)) {
        auto [mem, dst, val] = store->args<3>();
        emit_unsafe(mem);
        auto v_dst   = emit(dst);
        auto t_ptr   = convert(dst->type());
        auto pointee = mem::pointee(dst);
        auto size    = Lit::isa(core::op(core::trait::size, pointee));
        auto src     = async_copy_ && size && (*size == 4 || *size == 8 || *size == 16) ? global_source(val) : nullptr;
        if (!src) {
            auto v_val = emit(val);
            std::print(bb.body().emplace_back(), "store {} {}, {} {}", convert(pointee, false), v_val, t_ptr, v_dst);
            return std::string();
        }
        auto intrinsic = std::format("llvm.nvvm.cp.async.{}.shared.global.{}", *size == 16 ? "cg" : "ca", *size);
        declare("void @{}(ptr addrspace(3), ptr addrspace(1))", intrinsic);
        auto shared = bb.assign(name + ".s", "addrspacecast {} {} to ptr addrspace(3)", t_ptr, v_dst);
        auto global
            = bb.assign(name + ".g", "addrspacecast {} {} to ptr addrspace(1)", convert(src->type()), emit(src));
        std::print(bb.body().emplace_back(), "call void @{}(ptr addrspace(3) {}, ptr addrspace(1) {})", intrinsic,
                   shared, global);
        return std::string();
    } else if (auto ld = Axm::isa<gpu::ldmatrix>(def)) {
        auto [n, mem, ptr] = ld->args<3>();
        emit_unsafe(mem);
        auto count     = Lit::as(n);
        auto v_ptr     = emit(ptr);
        auto t_ptr     = convert(ptr->type());
        auto shared    = Lit::as(ld->decurry()->arg()) == 3
                           ? v_ptr
                           : bb.assign(name + ".s", "addrspacecast {} {} to ptr addrspace(3)", t_ptr, v_ptr);
        auto regs      = count == 1 ? "i32"s : std::format("{{ {} }}", fe::Join(std::vector(count, "i32"s), ", "));
        auto intrinsic = std::format("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x{}.b16.p3", count);
        declare("{} @{}(ptr addrspace(3))", regs, intrinsic);
        auto loaded = bb.assign(name + ".r", "call {} @{}(ptr addrspace(3) {})", regs, intrinsic, shared);
        if (count == 1) return bb.assign(name, "bitcast i32 {} to float", loaded);
        auto value = "undef"s;
        for (u64 i = 0; i != count; ++i) {
            auto reg = bb.assign(std::format("{}.i{}", name, i), "extractvalue {} {}, {}", regs, loaded, i);
            auto f   = bb.assign(std::format("{}.f{}", name, i), "bitcast i32 {} to float", reg);
            value = bb.assign(std::format("{}.v{}", name, i), "insertvalue [{} x float] {}, float {}, {}", count, value,
                              f, i);
        }
        return value;
    } else if (auto shfl = Axm::isa<gpu::shfl_xor>(def)) {
        auto [x, mask] = shfl->args<2>();
        auto v_x = emit(x), v_mask = emit(mask);
        declare("float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32)");
        return bb.assign(name, "call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {}, i32 {}, i32 31)", v_x,
                         v_mask);
    } else if (auto shfl = Axm::isa<gpu::shfl_idx>(def)) {
        auto [x, src] = shfl->args<2>();
        auto v_x = emit(x), v_src = emit(src);
        declare("float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32)");
        return bb.assign(name, "call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float {}, i32 {}, i32 31)", v_x, v_src);
    } else if (auto tf32 = Axm::isa<gpu::tf32_rna>(def)) {
        auto arg = emit(tf32->arg());
        declare("i32 @llvm.nvvm.f2tf32.rna(float)");
        auto bits = bb.assign(name + ".i", "call i32 @llvm.nvvm.f2tf32.rna(float {})", arg);
        return bb.assign(name, "bitcast i32 {} to float", bits);
    } else if (auto mma = Axm::isa<gpu::mma_tf32>(def)) {
        auto [a, b, c] = mma->args<3>();
        auto v_a = emit(a), v_b = emit(b), v_c = emit(c);
        auto lanes = [&](const std::string& v, u64 n, bool bits, std::string_view tag) {
            auto result = std::vector<std::string>();
            for (u64 i = 0; i != n; ++i) {
                auto f = bb.assign(std::format("{}.{}{}", name, tag, i), "extractvalue [{} x float] {}, {}", n, v, i);
                result.push_back(
                    bits ? "i32 " + bb.assign(std::format("{}.{}i{}", name, tag, i), "bitcast float {} to i32", f)
                         : "float " + f);
            }
            return result;
        };
        auto operands = lanes(v_a, 4, true, "a");
        for (auto&& op : lanes(v_b, 2, true, "b"))
            operands.push_back(op);
        for (auto&& op : lanes(v_c, 4, false, "c"))
            operands.push_back(op);
        auto regs = "{ float, float, float, float }"s;
        declare("{} @llvm.nvvm.mma.m16n8k8.row.col.tf32(i32, i32, i32, i32, i32, i32, float, float, float, float)",
                regs);
        auto d
            = bb.assign(name + ".d", "call {} @llvm.nvvm.mma.m16n8k8.row.col.tf32({})", regs, fe::Join(operands, ", "));
        auto value = "undef"s;
        for (u64 i = 0; i != 4; ++i) {
            auto f = bb.assign(std::format("{}.d{}", name, i), "extractvalue {} {}, {}", regs, d, i);
            value  = bb.assign(std::format("{}.v{}", name, i), "insertvalue [4 x float] {}, float {}, {}", value, f, i);
        }
        return value;
    } else if (auto commit = Axm::isa<gpu::async_commit>(def)) {
        emit_unsafe(commit->arg());
        if (async_copy_) {
            declare("void @llvm.nvvm.cp.async.commit.group()");
            std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.commit.group()");
        }
        return std::string();
    } else if (auto wait = Axm::isa<gpu::async_wait_group>(def)) {
        emit_unsafe(wait->arg());
        if (async_copy_) {
            declare("void @llvm.nvvm.cp.async.wait.group(i32)");
            std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
                       Lit::as(wait->decurry()->arg()));
        }
        return std::string();
    } else if (auto wait = Axm::isa<gpu::async_wait>(def)) {
        emit_unsafe(wait->arg());
        if (async_copy_) {
            declare("void @llvm.nvvm.cp.async.wait.all()");
            std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.wait.all()");
        }
        return std::string();
    } else if (Axm::isa<gpu::generic_mem>(def) || Axm::isa<gpu::global_mem>(def) || Axm::isa<gpu::panel_scope>(def)) {
        emit_unsafe(def->as<App>()->arg());
        return std::string();
    } else if (auto slot = Axm::isa<gpu::slot_shared>(def); slot && dynamic_slots_.contains(def)) {
        emit_unsafe(slot->arg());
        auto shared = bb.assign(name + ".s", "getelementptr inbounds i8, ptr addrspace(3) {}, i64 {}", dynamic_array_,
                                dynamic_slots_[def]);
        bb.assign(name, "addrspacecast ptr addrspace(3) {} to ptr", shared);
        return name;
    } else if (auto slot = Axm::isa<gpu::slot_shared>(def)) {
        emit_unsafe(slot->arg());
        auto pointee = convert(slot->decurry()->arg(), false);
        auto v_ptr   = "@" + def->unique_name() + ".slot";
        std::println(vars_decls_, "{} = internal addrspace(3) global {} undef, align 16", v_ptr, pointee);
        bb.assign(name, "addrspacecast {} addrspace(3)* {} to {}*", pointee, v_ptr, pointee);
        return name;
    } else if (auto tri = Axm::isa<math::tri>(def)) {
        auto arg       = emit(tri->arg());
        auto type      = convert(tri->arg()->type());
        auto func_name = ""s;
        switch (tri.id()) {
            case math::tri::ahff: func_name = "sin"; break;
            case math::tri::ahfF: func_name = "cos"; break;
            case math::tri::ahFf: func_name = "tan"; break;
            case math::tri::ahFF: break;
            case math::tri::aHff: func_name = "sinh"; break;
            case math::tri::aHfF: func_name = "cosh"; break;
            case math::tri::aHFf: func_name = "tanh"; break;
            case math::tri::aHFF: break;
            case math::tri::Ahff: func_name = "asin"; break;
            case math::tri::AhfF: func_name = "acos"; break;
            case math::tri::AhFf: func_name = "atan"; break;
            case math::tri::AhFF: break;
            case math::tri::AHff: func_name = "asinh"; break;
            case math::tri::AHfF: func_name = "acosh"; break;
            case math::tri::AHFf: func_name = "atanh"; break;
            case math::tri::AHFF: break;
        }
        if (func_name.empty()) fe::throwf("Trigonometric tag used by {} is currently unused", def);
        func_name                = func_name + ll::detail::math_suffix(tri->arg()->type());
        auto libdevice_func_name = "__nv_" + func_name;
        declare("{} @{}({})", type, libdevice_func_name, type);
        uses_libdevice = true;
        bb.assign(name, "call {} @{}({} {})", type, libdevice_func_name, type, arg);
        return name;
    } else if (auto exp = Axm::isa<math::exp>(def)) {
        auto arg       = emit(exp->arg());
        auto type      = convert(exp->arg()->type());
        auto func_name = ""s;
        switch (exp.id()) {
            case math::exp::lbb: func_name = "exp"; break;
            case math::exp::lbB: func_name = "exp2"; break;
            case math::exp::lBb: func_name = "exp10"; break;
            case math::exp::lBB: break;
            case math::exp::Lbb: func_name = "log"; break;
            case math::exp::LbB: func_name = "log2"; break;
            case math::exp::LBb: func_name = "log10"; break;
            case math::exp::LBB: break;
        }
        if (func_name.empty()) fe::throwf("Exponential tag used by {} is currently unused", def);
        func_name                = func_name + ll::detail::math_suffix(exp->arg()->type());
        auto libdevice_func_name = "__nv_" + func_name;
        declare("{} @{}({})", type, libdevice_func_name, type);
        uses_libdevice = true;
        bb.assign(name, "call {} @{}({} {})", type, libdevice_func_name, type, arg);
        return name;
    }
    return std::nullopt;
}

void emit_host(World& world,
               std::ostream& ostream,
               std::optional<std::string> device_fatbin_file,
               std::map<std::string, u64> dynamic_shared,
               ll::Emitter::Rt rt) {
    HostEmitter emitter(world, ostream, device_fatbin_file, std::move(dynamic_shared));
    emitter.rt_mode(rt);
    // Same one-liner the `ll` backend uses; each backend just names its own runtime module.
    if (rt == ll::Emitter::Rt::embed) emitter.load_rt_module("nvptx_rt.ll");
    emitter.run();
}

DeviceEmitFlags emit_device(World& world, std::ostream& ostream, bool async_copy) {
    DeviceEmitter emitter(world, ostream, async_copy);
    emitter.run();

    return DeviceEmitFlags{
        .uses_libdevice = emitter.is_using_libdevice(),
        .dynamic_shared = emitter.dynamic_shared(),
    };
}

} // namespace mim::plug::nvptx
