#include "mim/plug/nvptx/phase/nvptx.h"

#include <algorithm>
#include <format>

#include <fe/format.h>
#include <fe/log.h>
#include <fe/sys.h>

#include <mim/driver.h>
#include <mim/lam.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/math/math.h>
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

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_device_emitter", ostream) {
        simd_ = false;
    }

    void start() final;

    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

    bool is_using_libdevice() const { return uses_libdevice; }
    bool is_using_wgmma() const { return uses_wgmma_; }
    const std::map<std::string, u64>& dynamic_shared() const { return dynamic_shared_; }
    const std::string& get_extra_flags() const { return extra_flags; }

private:
    /// Device slots live in a module-scope global (or the dynamic shared arena when > 48 KiB).
    std::string emit_slot(ll::BB& bb, const App* app, const Def* pointee, const Def* addr_space) override {
        auto as = Lit::as(addr_space);
        if (as == 3) {
            if (auto it = dynamic_slots_.find(app); it != dynamic_slots_.end()) {
                auto name     = id(app);
                auto byte_ptr = bb.assign(name + ".dyn",
                                          "getelementptr inbounds [0 x i8], [0 x i8] addrspace(3)* {}, i64 0, i64 {}",
                                          dynamic_array_, it->second);
                return bb.assign(name + ".slot", "bitcast i8 addrspace(3)* {} to {} addrspace(3)*", byte_ptr,
                                 convert(pointee, false));
            }
            auto size  = Lit::isa(core::op(core::trait::size, pointee)).value_or(0);
            auto align = (kernel_uses_wgmma_ && size >= 1024) ? 1024 : 16;
            auto v_ptr = "@" + app->unique_name() + ".slot";
            std::println(vars_decls_, "{} = internal addrspace(3) global {} undef, align {}", v_ptr,
                         convert(pointee, false), align);
            return v_ptr;
        }
        auto v_ptr = "@" + app->unique_name() + ".slot";
        std::println(vars_decls_, "{} = internal addrspace({}) global {} undef, align 16", v_ptr, addr_space,
                     convert(pointee, false));
        return v_ptr;
    }

    static void
    inline_asm(ll::BB& bb, std::string_view text, std::string_view constraints, const std::vector<std::string>& args) {
        auto clobber = constraints.empty() ? "~{memory}"s : std::format("{},~{{memory}}", constraints);
        std::print(bb.body().emplace_back(), "call void asm sideeffect \"{}\", \"{}\"({})", text, clobber,
                   fe::Join(args, ", "));
    }

    static void
    leaf_paths(const Def* type, const std::string& prefix, std::vector<std::pair<std::string, const Def*>>& paths) {
        if (auto arr = type->isa<Arr>(); arr && Lit::isa(arr->arity()))
            for (u64 i = 0, e = Lit::as(arr->arity()); i != e; ++i)
                leaf_paths(arr->elem(), std::format("{}, {}", prefix, i), paths);
        else if (auto sigma = type->isa<Sigma>())
            for (size_t i = 0; i != sigma->num_ops(); ++i)
                leaf_paths(sigma->op(i), std::format("{}, {}", prefix, i), paths);
        else
            paths.emplace_back(prefix, type);
    }

    LamSet kernels_;
    DefMap<u64> dynamic_slots_;
    std::string dynamic_array_;
    std::map<std::string, u64> dynamic_shared_;
    bool kernel_uses_wgmma_ = false;
    bool uses_wgmma_        = false;
    bool uses_libdevice     = false;
    std::string extra_flags;
};

namespace {
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
    } else if (auto dmap = Axm::isa<gpu::device_tensor_map>(def)) {
        auto [m1, base, rows, cols, box_rows, box_cols] = dmap->args<6>();
        emit_unsafe(m1);
        auto elem  = dmap->decurry()->decurry()->arg();
        auto dtype = math::isa_bf16(elem) ? 9 : math::isa_f(elem) == 16 ? 6 : math::isa_f(elem) == 32 ? 7 : -1;
        if (dtype < 0) def->blame(MIM_NVPTX_BE "a tensor map of `{}` elements is not supported", elem).bail();
        auto size    = Lit::as(core::op(core::trait::size, elem));
        auto v_map   = bb.assign(name + ".host", "alloca [16 x i64], align 64");
        auto dims    = bb.assign(name + ".dims", "alloca [2 x i64]");
        auto strides = bb.assign(name + ".strides", "alloca i64");
        auto box     = bb.assign(name + ".box", "alloca [2 x i32]");
        auto unit    = bb.assign(name + ".unit", "alloca [2 x i32]");
        auto at      = [&](const std::string& arr, std::string_view t, int i) {
            return bb.assign(std::format("{}.{}", arr, i), "getelementptr [2 x {}], ptr {}, i64 0, i64 {}", t, arr, i);
        };
        auto store = [&](std::string_view t, const std::string& v, const std::string& ptr) {
            std::print(bb.body().emplace_back(), "store {} {}, ptr {}", t, v, ptr);
        };
        auto v_cols = emit(cols), v_rows = emit(rows);
        store("i64", v_cols, at(dims, "i64", 0));
        store("i64", v_rows, at(dims, "i64", 1));
        store("i64", bb.assign(name + ".stride", "mul i64 {}, {}", v_cols, size), strides);
        store("i32", bb.assign(name + ".bc", "trunc i64 {} to i32", emit(box_cols)), at(box, "i32", 0));
        store("i32", bb.assign(name + ".br", "trunc i64 {} to i32", emit(box_rows)), at(box, "i32", 1));
        store("i32", "1", at(unit, "i32", 0));
        store("i32", "1", at(unit, "i32", 1));
        auto v_base = emit(base);
        auto global
            = convert(base->type()) == "i64" ? bb.assign(name + ".base", "inttoptr i64 {} to ptr", v_base) : v_base;
        declare("i32 @cuTensorMapEncodeTiled(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32)");
        auto res = bb.assign(name + ".res",
                             "call i32 @cuTensorMapEncodeTiled(ptr {}, i32 {}, i32 2, ptr {}, ptr {}, ptr {}, ptr {}, "
                             "ptr {}, i32 0, i32 3, i32 2, i32 0)",
                             v_map, dtype, global, dims, strides, box, unit);
        emit_cu_error_handling(bb, res);
        declare("i32 @{}(ptr, i64)", Cu_Mem_Alloc);
        auto device = bb.assign(name + ".dev", "alloca i64");
        emit_cu_error_handling(bb, bb.assign(name + ".alloc", "call i32 @{}(ptr {}, i64 128)", Cu_Mem_Alloc, device));
        auto v_dev = bb.assign(name, "load i64, ptr {}", device);
        declare("i32 @{}(i64, ptr, i64)", Cu_Memcpy_Htod);
        emit_cu_error_handling(
            bb, bb.assign(name + ".copy", "call i32 @{}(i64 {}, ptr {}, i64 128)", Cu_Memcpy_Htod, v_dev, v_map));
        return v_dev;
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
        if (auto it = dynamic_shared_.find(id(lam).substr(1)); it != dynamic_shared_.end())
            shared_mem_bytes = std::max(shared_mem_bytes, it->second);

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
    kernel_uses_wgmma_ = false;
    DefVec shared_slots;
    DefSet seen;
    for (DefVec todo{kernel}; !todo.empty();) {
        auto d = todo.back();
        todo.pop_back();
        if (!seen.emplace(d).second) continue;
        if (Axm::isa<gpu::wgmma_f16>(d) || Axm::isa<gpu::wgmma_bf16>(d) || Axm::isa<gpu::tma_load_if>(d)) {
            kernel_uses_wgmma_ = true;
            uses_wgmma_        = true;
        }
        if (auto mslot = Axm::isa<mem::mslot>(d)) {
            auto [Ta, _]             = mslot->uncurry_args<2>();
            auto [__, addr_space]    = Ta->projs<2>();
            if (Lit::isa(addr_space) == 3) shared_slots.push_back(mslot);
        }
        for (auto dep : d->deps())
            todo.push_back(dep);
    }
    u64 total_shared = 0;
    for (auto slot : shared_slots) {
        auto pointee = slot->as<App>()->uncurry_args<2>()[0]->proj(2, 0);
        auto sz      = Lit::isa(core::op(core::trait::size, pointee)).value_or(0);
        u64 align    = (kernel_uses_wgmma_ && sz >= 1024) ? 1024 : 16;
        total_shared = (total_shared + align - 1) / align * align;
        dynamic_slots_[slot] = total_shared;
        total_shared += (sz + 15) / 16 * 16;
    }
    if (total_shared > 48 * 1024) {
        auto kname             = id(kernel).substr(1);
        dynamic_array_         = "@" + kname + "_dynamic_shared";
        dynamic_shared_[kname] = total_shared;
        std::println(vars_decls_, "{} = external addrspace(3) global [0 x i8], align {}", dynamic_array_,
                     kernel_uses_wgmma_ ? 1024 : 16);
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
        std::println(vars_decls_, "{} = internal addrspace({}) global {} undef", name, a, convert(T, false));
    }

    return kernel->unique_name();
}

std::optional<std::string> DeviceEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        declare("void @llvm.nvvm.barrier0()");

        emit_unsafe(sync_work_items->arg(2, 0));
        emit_unsafe(sync_work_items->arg(2, 1));
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.barrier0()");
        return name;
    } else if (auto shfl = Axm::isa<gpu::warp_shfl_xor>(def)) {
        auto [m_def, val_def, off_def] = shfl->args<3>();
        emit_unsafe(m_def);
        auto val = emit(val_def);
        auto off = emit(off_def);
        auto ty  = convert(val_def->type());
        if (ty == "float") {
            declare("float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32)");
            return bb.assign(name, "call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {}, i32 {}, i32 31)", val,
                             off);
        } else if (ty == "i32") {
            declare("i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)");
            return bb.assign(name, "call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {}, i32 {}, i32 31)", val, off);
        } else if (ty == "double" || ty == "i64") {
            declare("i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)");
            auto bits  = ty == "double" ? bb.assign(name + ".bits", "bitcast double {} to i64", val) : val;
            auto lo    = bb.assign(name + ".lo", "trunc i64 {} to i32", bits);
            auto hi64  = bb.assign(name + ".hi64", "lshr i64 {}, 32", bits);
            auto hi    = bb.assign(name + ".hi", "trunc i64 {} to i32", hi64);
            auto slo   = bb.assign(name + ".slo",
                                   "call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {}, i32 {}, i32 31)", lo, off);
            auto shi   = bb.assign(name + ".shi",
                                   "call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {}, i32 {}, i32 31)", hi, off);
            auto zlo   = bb.assign(name + ".zlo", "zext i32 {} to i64", slo);
            auto zhi   = bb.assign(name + ".zhi", "zext i32 {} to i64", shi);
            auto shi64 = bb.assign(name + ".shi64", "shl i64 {}, 32", zhi);
            auto comb  = bb.assign(name + ".comb", "or i64 {}, {}", shi64, zlo);
            if (ty == "double") return bb.assign(name, "bitcast i64 {} to double", comb);
            return comb;
        } else if (ty == "i1" || ty == "i8" || ty == "i16") {
            declare("i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)");
            auto ext = bb.assign(name + ".ext", "zext {} {} to i32", ty, val);
            auto sh  = bb.assign(name + ".sh", "call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {}, i32 {}, i32 31)",
                                 ext, off);
            return bb.assign(name, "trunc i32 {} to {}", sh, ty);
        } else {
            fe::throwf(MIM_NVPTX_BE "`gpu.warp_shfl_xor` does not support type `{}`", ty);
        }
    } else if (auto vload = Axm::isa<gpu::vload4>(def)) {
        auto [m_def, ptr_def] = vload->args<2>();
        emit_unsafe(m_def);
        auto v_ptr   = emit(ptr_def);
        auto ptr_ty  = Axm::expect<mem::Ptr>(ptr_def->type(), "a `mem.Ptr`");
        auto [T, as] = ptr_ty->args<2>();
        auto t_elem  = convert(T, false);
        auto t_ptr   = convert(ptr_ty);
        auto as_lit  = Lit::expect(as, "an address space");
        auto t_vptr  = std::format("<4 x {}> addrspace({})*", t_elem, as_lit);
        auto vp      = bb.assign(name + ".vp", "bitcast {} {} to {}", t_ptr, v_ptr, t_vptr);
        auto vec     = bb.assign(name + ".vec", "load <4 x {}>, {} {}, align 16", t_elem, t_vptr, vp);
        auto arr_ty  = std::format("[4 x {}]", t_elem);
        std::string cur = "undef";
        for (int i = 0; i < 4; ++i) {
            auto el  = bb.assign(std::format("{}.e{}", name, i), "extractelement <4 x {}> {}, i32 {}", t_elem, vec, i);
            auto dst = i == 3 ? name : std::format("{}.a{}", name, i);
            cur      = bb.assign(dst, "insertvalue {} {}, {} {}, {}", arr_ty, cur, t_elem, el, i);
        }
        return cur;
    } else if (auto vstore = Axm::isa<gpu::vstore4>(def)) {
        auto [m_def, ptr_def, val_def] = vstore->args<3>();
        emit_unsafe(m_def);
        auto v_ptr   = emit(ptr_def);
        auto v_val   = emit(val_def);
        auto ptr_ty  = Axm::expect<mem::Ptr>(ptr_def->type(), "a `mem.Ptr`");
        auto [T, as] = ptr_ty->args<2>();
        auto t_elem  = convert(T, false);
        auto t_ptr   = convert(ptr_ty);
        auto as_lit  = Lit::expect(as, "an address space");
        auto arr_ty  = std::format("[4 x {}]", t_elem);
        auto t_vptr  = std::format("<4 x {}> addrspace({})*", t_elem, as_lit);
        std::string vec = "undef";
        for (int i = 0; i < 4; ++i) {
            auto el = bb.assign(std::format("{}.e{}", name, i), "extractvalue {} {}, {}", arr_ty, v_val, i);
            vec     = bb.assign(std::format("{}.v{}", name, i), "insertelement <4 x {}> {}, {} {}, i32 {}", t_elem, vec,
                                t_elem, el, i);
        }
        auto vp = bb.assign(name + ".vp", "bitcast {} {} to {}", t_ptr, v_ptr, t_vptr);
        std::print(bb.body().emplace_back(), "store <4 x {}> {}, {} {}, align 16", t_elem, vec, t_vptr, vp);
        return std::string();
    } else if (auto vstore2 = Axm::isa<gpu::vstore2>(def)) {
        auto [m_def, ptr_def, val_def] = vstore2->args<3>();
        emit_unsafe(m_def);
        auto v_ptr   = emit(ptr_def);
        auto v_val   = emit(val_def);
        auto ptr_ty  = Axm::expect<mem::Ptr>(ptr_def->type(), "a `mem.Ptr`");
        auto [T, as] = ptr_ty->args<2>();
        auto t_elem  = convert(T, false);
        auto t_ptr   = convert(ptr_ty);
        auto as_lit  = Lit::expect(as, "an address space");
        auto arr_ty  = std::format("[2 x {}]", t_elem);
        auto t_vptr  = std::format("<2 x {}> addrspace({})*", t_elem, as_lit);
        std::string vec = "undef";
        for (int i = 0; i < 2; ++i) {
            auto el = bb.assign(std::format("{}.e{}", name, i), "extractvalue {} {}, {}", arr_ty, v_val, i);
            vec     = bb.assign(std::format("{}.v{}", name, i), "insertelement <2 x {}> {}, {} {}, i32 {}", t_elem, vec,
                                t_elem, el, i);
        }
        auto vp = bb.assign(name + ".vp", "bitcast {} {} to {}", t_ptr, v_ptr, t_vptr);
        std::print(bb.body().emplace_back(), "store <2 x {}> {}, {} {}, align 8", t_elem, vec, t_vptr, vp);
        return std::string();
    } else if (auto cp = Axm::isa<gpu::cp_async_cg_16>(def)) {
        auto [m1_def, m3_def, dst_def, src_def] = cp->args<4>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        auto v_dst = emit(dst_def);
        auto v_src = emit(src_def);
        auto t_dst = convert(dst_def->type());
        auto t_src = convert(src_def->type());
        auto p3    = bb.assign(name + ".p3", "bitcast {} {} to i8 addrspace(3)*", t_dst, v_dst);
        auto p1    = bb.assign(name + ".p1", "bitcast {} {} to i8 addrspace(1)*", t_src, v_src);
        declare("void @llvm.nvvm.cp.async.cg.shared.global.16(i8 addrspace(3)*, i8 addrspace(1)*)");
        std::print(bb.body().emplace_back(),
                   "call void @llvm.nvvm.cp.async.cg.shared.global.16(i8 addrspace(3)* {}, i8 addrspace(1)* {})", p3,
                   p1);
        return name;
    } else if (auto z16 = Axm::isa<gpu::cp_async_zero_16>(def)) {
        auto [m3_def, dst_def] = z16->args<2>();
        emit_unsafe(m3_def);
        auto v_dst = emit(dst_def);
        auto t_dst = convert(dst_def->type());
        auto vp    = bb.assign(name + ".vp", "bitcast {} {} to <4 x i32> addrspace(3)*", t_dst, v_dst);
        std::print(bb.body().emplace_back(),
                   "store <4 x i32> zeroinitializer, <4 x i32> addrspace(3)* {}, align 16", vp);
        return std::string();
    } else if (auto commit = Axm::isa<gpu::cp_async_commit>(def)) {
        emit_unsafe(commit->arg(2, 0));
        emit_unsafe(commit->arg(2, 1));
        declare("void @llvm.nvvm.cp.async.commit.group()");
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.commit.group()");
        return name;
    } else if (auto wait = Axm::isa<gpu::cp_async_wait_group>(def)) {
        emit_unsafe(wait->arg(2, 0));
        emit_unsafe(wait->arg(2, 1));
        auto n_pending = Lit::expect(wait->decurry()->arg(), "a static cp.async wait count");
        declare("void @llvm.nvvm.cp.async.wait.group(i32)");
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.cp.async.wait.group(i32 {})", n_pending);
        return name;
    } else if (auto ld = Axm::isa<gpu::ldmatrix_b16>(def)) {
        auto [n_def, trans_def, m1_def, m3_def, ptr_def] = ld->args<5>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        auto count     = Lit::expect(n_def, "a static ldmatrix count");
        auto trans     = Lit::as<bool>(trans_def);
        auto v_ptr     = emit(ptr_def);
        auto t_ptr     = convert(ptr_def->type());
        auto elem      = convert(ld->decurry()->arg(2, 0), false);
        auto p3        = bb.assign(name + ".ptr", "bitcast {} {} to i8 addrspace(3)*", t_ptr, v_ptr);
        auto regs      = count == 1 ? "i32"s : std::format("{{ {} }}", fe::Join(std::vector(count, "i32"s), ", "));
        auto intrinsic = std::format("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x{}{}.b16.p3", count, trans ? ".trans" : "");
        declare("{} @{}(i8 addrspace(3)*)", regs, intrinsic);
        auto loaded = bb.assign(name + ".r", "call {} @{}(i8 addrspace(3)* {})", regs, intrinsic, p3);
        auto t_val  = count == 1 ? std::format("[2 x {}]", elem) : std::format("[{} x [2 x {}]]", count, elem);
        std::string value = "undef";
        for (u64 i = 0; i != count; ++i) {
            auto reg  = count == 1
                          ? loaded
                          : bb.assign(std::format("{}.i{}", name, i), "extractvalue {} {}, {}", regs, loaded, i);
            auto pair = bb.assign(std::format("{}.p{}", name, i), "bitcast i32 {} to <2 x {}>", reg, elem);
            for (u64 h = 0; h != 2; ++h) {
                auto x   = bb.assign(std::format("{}.e{}_{}", name, i, h), "extractelement <2 x {}> {}, i32 {}", elem,
                                     pair, h);
                auto dst = std::format("{}.v{}_{}", name, i, h);
                value = count == 1 ? bb.assign(dst, "insertvalue {} {}, {} {}, {}", t_val, value, elem, x, h)
                                   : bb.assign(dst, "insertvalue {} {}, {} {}, {}, {}", t_val, value, elem, x, i, h);
            }
        }
        return bb.assign(name, "insertvalue {{ {} }} undef, {} {}, 0", t_val, t_val, value);
    } else if (Axm::isa<gpu::mma_m16n8k16_f16>(def) || Axm::isa<gpu::mma_m16n8k16_bf16>(def)) {
        bool bf              = Axm::isa<gpu::mma_m16n8k16_bf16>(def) != nullptr;
        auto [m1_def, a, b, c] = def->as<App>()->args<4>();
        emit_unsafe(m1_def);
        auto v_a  = emit(a), v_b = emit(b), v_c = emit(c);
        auto elem = bf ? "bfloat"s : "half"s;
        auto pack_pairs = [&](const std::string& v, u64 n, std::string_view tag) {
            std::vector<std::string> res;
            for (u64 i = 0; i != n; ++i) {
                std::string pair = std::format("<2 x {}> poison", elem);
                for (u64 j = 0; j != 2; ++j) {
                    auto x = bb.assign(std::format("{}.{}{}_{}", name, tag, i, j),
                                       "extractvalue [{} x [2 x {}]] {}, {}, {}", n, elem, v, i, j);
                    pair   = std::format("<2 x {}> ", elem)
                           + bb.assign(std::format("{}.{}{}v{}", name, tag, i, j), "insertelement {}, {} {}, i32 {}",
                                       pair, elem, x, j);
                }
                if (bf) pair = "i32 " + bb.assign(std::format("{}.{}{}i", name, tag, i), "bitcast {} to i32", pair);
                res.push_back(pair);
            }
            return res;
        };
        auto operands = pack_pairs(v_a, 4, "a");
        for (auto&& op : pack_pairs(v_b, 2, "b"))
            operands.push_back(op);
        for (u64 i = 0; i != 4; ++i)
            operands.push_back("float "
                               + bb.assign(std::format("{}.c{}", name, i), "extractvalue [4 x float] {}, {}", v_c, i));
        auto regs   = "{ float, float, float, float }"s;
        auto intr   = bf ? "llvm.nvvm.mma.m16n8k16.row.col.bf16"s : "llvm.nvvm.mma.m16n8k16.row.col.f32.f32"s;
        auto reg_ty = bf ? "i32"s : "<2 x half>"s;
        declare("{} @{}({}, {}, {}, {}, {}, {}, float, float, float, float)", regs, intr, reg_ty, reg_ty, reg_ty,
                reg_ty, reg_ty, reg_ty);
        auto d            = bb.assign(name + ".d", "call {} @{}({})", regs, intr, fe::Join(operands, ", "));
        std::string value = "undef";
        for (u64 i = 0; i != 4; ++i) {
            auto f   = bb.assign(std::format("{}.d{}", name, i), "extractvalue {} {}, {}", regs, d, i);
            auto dst = i == 3 ? name : std::format("{}.v{}", name, i);
            value    = bb.assign(dst, "insertvalue [4 x float] {}, float {}, {}", value, f, i);
        }
        return value;
    } else if (auto addr = Axm::isa<gpu::shared_addr>(def)) {
        auto ptr   = addr->arg();
        auto v_ptr = emit(ptr);
        auto t_ptr = convert(ptr->type());
        auto p3    = bb.assign(name + ".p3", "bitcast {} {} to i8 addrspace(3)*", t_ptr, v_ptr);
        return bb.assign(name, "ptrtoint i8 addrspace(3)* {} to i32", p3);
    } else if (auto mi = Axm::isa<gpu::mbar_init_if>(def)) {
        auto [m1_def, m3_def, bar, count, issue] = mi->args<5>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        auto pred = bb.assign(name + ".p", "zext i1 {} to i32", emit(issue));
        inline_asm(bb, "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q mbarrier.init.shared::cta.b64 [$0], $1; }", "r,r,r",
                   {"i32 " + emit(bar), "i32 " + emit(count), "i32 " + pred});
        return name;
    } else if (auto me = Axm::isa<gpu::mbar_expect_tx_if>(def)) {
        auto [m1_def, m3_def, bar, bytes, issue] = me->args<5>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        auto pred = bb.assign(name + ".p", "zext i1 {} to i32", emit(issue));
        inline_asm(
            bb, "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1; }",
            "r,r,r", {"i32 " + emit(bar), "i32 " + emit(bytes), "i32 " + pred});
        return name;
    } else if (auto mw = Axm::isa<gpu::mbar_wait_at>(def)) {
        auto [m1_def, m3_def, bar, parity] = mw->args<4>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        inline_asm(
            bb, "{ .reg .pred p; W${:uid}: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra W${:uid}; }",
            "r,r", {"i32 " + emit(bar), "i32 " + emit(parity)});
        return name;
    } else if (auto tma = Axm::isa<gpu::tma_load_if>(def)) {
        auto [m1_def, m3_def, dst, map, col, row, bar, issue] = tma->args<8>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        auto v_map = emit(map);
        auto t_map = convert(map->type());
        auto gm    = bb.assign(name + ".gm", "bitcast {} {} to i8 addrspace(1)*", t_map, v_map);
        auto map_i = bb.assign(name + ".map", "ptrtoint i8 addrspace(1)* {} to i64", gm);
        auto pred  = bb.assign(name + ".p", "zext i1 {} to i32", emit(issue));
        inline_asm(bb,
                   "{ .reg .pred q; setp.ne.b32 q, $5, 0; @q "
                   "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], "
                   "[$4]; }",
                   "r,l,r,r,r,r",
                   {"i32 " + emit(dst), "i64 " + map_i, "i32 " + emit(col), "i32 " + emit(row), "i32 " + emit(bar),
                    "i32 " + pred});
        return name;
    } else if (auto wf = Axm::isa<gpu::wgmma_fence>(def)) {
        emit_unsafe(wf->arg(2, 0));
        emit_unsafe(wf->arg(2, 1));
        declare("void @llvm.nvvm.wgmma.fence.sync.aligned()");
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.wgmma.fence.sync.aligned()");
        return name;
    } else if (auto wc = Axm::isa<gpu::wgmma_commit>(def)) {
        emit_unsafe(wc->arg(2, 0));
        emit_unsafe(wc->arg(2, 1));
        declare("void @llvm.nvvm.wgmma.commit_group.sync.aligned()");
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.wgmma.commit_group.sync.aligned()");
        return name;
    } else if (Axm::isa<gpu::wgmma_f16>(def) || Axm::isa<gpu::wgmma_bf16>(def)) {
        auto [nt, m1_def, m3_def, a, b, trans_b, acc] = def->as<App>()->args<7>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        bool bf    = Axm::isa<gpu::wgmma_bf16>(def) != nullptr;
        auto n     = Lit::as(nt);
        auto regs  = 4 * n;
        auto v_a   = emit(a);
        auto v_b   = emit(b);
        auto v_acc = emit(acc);
        auto t_acc = convert(acc->type());
        auto ty    = bf ? "bf16"s : "f16"s;
        std::vector<std::string> ins;
        for (u64 j = 0; j != n; ++j)
            for (u64 e = 0; e != 4; ++e)
                ins.push_back("float "
                              + bb.assign(std::format("{}.c{}_{}", name, j, e), "extractvalue {} {}, {}, {}", t_acc,
                                          v_acc, j, e));
        std::vector<std::string> outs;
        for (u64 i = 0; i != regs; ++i)
            outs.push_back(std::format("${}", i));
        std::vector<std::string> constraints(regs, "=f");
        for (u64 i = 0; i != regs; ++i)
            constraints.push_back(std::to_string(i));
        ins.push_back("i64 " + v_a);
        constraints.push_back("l");
        ins.push_back("i64 " + v_b);
        constraints.push_back("l");
        auto flags    = std::format("1, 1, 1, 0, {}", Lit::as<bool>(trans_b) ? 1 : 0);
        auto asm_text = std::format("wgmma.mma_async.sync.aligned.m64n{}k16.f32.{}.{} {{{}}}, ${}, ${}, {};", 8 * n, ty,
                                    ty, fe::Join(outs, ", "), 2 * regs, 2 * regs + 1, flags);
        auto t_ret    = std::format("{{ {} }}", fe::Join(std::vector(regs, "float"s), ", "));
        auto d        = bb.assign(name + ".d", "call {} asm sideeffect \"{}\", \"{}\"({})", t_ret, asm_text,
                                  fe::Join(constraints, ","), fe::Join(ins, ", "));
        std::string value = "undef";
        for (u64 j = 0; j != n; ++j)
            for (u64 e = 0; e != 4; ++e) {
                auto f = bb.assign(std::format("{}.d{}_{}", name, j, e), "extractvalue {} {}, {}", t_ret, d, 4 * j + e);
                value  = bb.assign(std::format("{}.v{}_{}", name, j, e), "insertvalue {} {}, float {}, {}, {}", t_acc,
                                   value, f, j, e);
            }
        return bb.assign(name, "insertvalue {{ {} }} undef, {} {}, 0", t_acc, t_acc, value);
    } else if (auto wwg = Axm::isa<gpu::wgmma_wait_group>(def)) {
        auto [m1_def, m3_def, n_def] = wwg->args<3>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        declare("void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64)");
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64 {})",
                   Lit::as(n_def));
        return name;
    } else if (auto ww = Axm::isa<gpu::wgmma_wait>(def)) {
        auto [m1_def, m3_def, n_def, x] = ww->args<4>();
        emit_unsafe(m1_def);
        emit_unsafe(m3_def);
        auto v_x = emit(x);
        auto n   = Lit::as(n_def);
        declare("void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64)");
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64 {})", n);
        auto t_x = convert(x->type());
        std::vector<std::pair<std::string, const Def*>> paths;
        leaf_paths(x->type(), "", paths);
        auto value = v_x;
        for (size_t i = 0; i != paths.size(); ++i) {
            auto& [path, leaf] = paths[i];
            auto t             = convert(leaf);
            auto reg           = t == "float" ? "f" : t == "double" ? "d" : t == "half" || t == "bfloat" ? "h" : "r";
            auto f             = bb.assign(std::format("{}.w{}", name, i), "extractvalue {} {}{}", t_x, v_x, path);
            auto g             = bb.assign(std::format("{}.f{}", name, i), "call {} asm sideeffect \"\", \"={},0\"({} {})",
                                           t, reg, t, f);
            value              = bb.assign(std::format("{}.x{}", name, i), "insertvalue {} {}, {} {}{}", t_x, value, t, g,
                                           path);
        }
        return bb.assign(name, "insertvalue {{ {} }} undef, {} {}, 0", t_x, t_x, value);
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

DeviceEmitFlags emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();

    return DeviceEmitFlags{
        .uses_libdevice = emitter.is_using_libdevice(),
        .uses_wgmma     = emitter.is_using_wgmma(),
        .dynamic_shared = emitter.dynamic_shared(),
    };
}

} // namespace mim::plug::nvptx
