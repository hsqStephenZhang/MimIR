// kbench16 <cubin> <kernel> <groups> <items> <f16|bf16>[f32[t]] [iters] [n] [smem]: times a generated n³ 16-bit GEMM kernel back to back.
// The kernel takes its three device buffers as one by-value array, row-major, C in the input type or with `f32` in F32, `t` reading B as Bᵀ;
// the operand order is found by checking the result.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include <cuda.h>

#define CHECK(x)                                                                         \
    do {                                                                                 \
        if (auto r = (x); r != CUDA_SUCCESS) {                                           \
            std::fprintf(stderr, "%s failed: %d\n", #x, int(r));                          \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

struct Bf16 {
    unsigned short bits;
    Bf16() = default;
    explicit Bf16(float f) {
        unsigned u;
        std::memcpy(&u, &f, 4);
        bits = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
    explicit operator float() const {
        unsigned u = unsigned(bits) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
};

template<class H, class C> int run(int argc, char** argv, bool bt = false) {
    int groups = std::atoi(argv[3]), items = std::atoi(argv[4]), iters = argc > 6 ? std::atoi(argv[6]) : 300;
    size_t N      = argc > 7 ? std::atoi(argv[7]) : 1024;
    unsigned smem = argc > 8 ? std::atoi(argv[8]) : 0;
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CUmodule mod;
    CHECK(cuModuleLoad(&mod, argv[1]));
    CUfunction fn;
    CHECK(cuModuleGetFunction(&fn, mod, argv[2]));
    if (smem) CHECK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));

    std::vector<H> a(N * N), b(N * N);
    std::vector<C> c(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = H(float(i % 97) / 97.0f - 0.5f);
        b[i] = H(float(i % 89) / 89.0f - 0.5f);
    }
    std::array<CUdeviceptr, 3> buf;
    for (auto& p : buf) CHECK(cuMemAlloc(&p, N * N * 4));
    void* params[] = {buf.data()};
    auto check = [&](int ia, int ib, int ic, bool report) {
        CHECK(cuMemcpyHtoD(buf[ia], a.data(), N * N * sizeof(H)));
        CHECK(cuMemcpyHtoD(buf[ib], b.data(), N * N * sizeof(H)));
        CHECK(cuMemsetD8(buf[ic], 0, N * N * sizeof(C)));
        CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, smem, nullptr, params, nullptr));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(c.data(), buf[ic], N * N * sizeof(C)));
        int bad = 0;
        for (size_t s = 0; s < 256; ++s) {
            size_t i = (s * 131 + 7) % N, j = (s * 977 + 3) % N;
            double ref = 0;
            for (size_t k = 0; k < N; ++k) ref += double(float(a[i * N + k])) * double(float(b[bt ? j * N + k : k * N + j]));
            double got = float(c[i * N + j]);
            if (std::fabs(ref - got) > 2e-2 * (1 + std::fabs(ref))) {
                if (report && bad < 3) std::printf("C[%zu][%zu] = %g want %g\n", i, j, got, ref);
                ++bad;
            }
        }
        return bad;
    };
    int order[3] = {0, 1, 2}, bad = check(0, 1, 2, false);
    for (int ia = 0; ia < 3 && bad; ++ia)
        for (int ib = 0; ib < 3 && bad; ++ib)
            if (ib != ia && !(bad = check(ia, ib, 3 - ia - ib, false))) order[0] = ia, order[1] = ib, order[2] = 3 - ia - ib;
    if (bad) {
        check(0, 1, 2, true);
        std::printf("FAIL\n");
        return 1;
    }
    check(order[0], order[1], order[2], false);
    for (int i = 0; i < 20; ++i) CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, smem, nullptr, params, nullptr));
    CUevent start, stop;
    CHECK(cuEventCreate(&start, 0));
    CHECK(cuEventCreate(&stop, 0));
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i) {
        CHECK(cuEventRecord(start, nullptr));
        CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, smem, nullptr, params, nullptr));
        CHECK(cuEventRecord(stop, nullptr));
        CHECK(cuEventSynchronize(stop));
        float t;
        CHECK(cuEventElapsedTime(&t, start, stop));
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    double med = ms[ms.size() / 2];
    std::printf("OK median %.4f ms  min %.4f ms  %.1f TFLOP/s\n", med, ms[0], 2.0 * N * N * N / (med * 1e-3) / 1e12);
    return 0;
}

int main(int argc, char** argv) {
    auto ty = std::string_view(argv[5]);
    if (ty == "bf16") return run<Bf16, Bf16>(argc, argv);
    if (ty == "bf16f32") return run<Bf16, float>(argc, argv);
    if (ty == "f16f32") return run<_Float16, float>(argc, argv);
    if (ty == "f16f32t") return run<_Float16, float>(argc, argv, true);
    if (ty == "bf16f32t") return run<Bf16, float>(argc, argv, true);
    return run<_Float16, _Float16>(argc, argv);
}
