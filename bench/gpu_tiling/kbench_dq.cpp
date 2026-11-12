// kbench_dq <cubin> <kernel> <groups> <items> [iters] [n] [smem]: times a generated n³ dequant GEMM
// `C = A · (W ∘ s)` with A F32, W I8 and a per-column F32 scale; the operand order is found by checking the result.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda.h>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (auto r = (x); r != CUDA_SUCCESS) {                                                                         \
            std::fprintf(stderr, "%s failed: %d\n", #x, int(r));                                                       \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

int main(int argc, char** argv) {
    int groups = std::atoi(argv[3]), items = std::atoi(argv[4]), iters = argc > 5 ? std::atoi(argv[5]) : 300;
    size_t N      = argc > 6 ? std::atoi(argv[6]) : 1024;
    unsigned smem = argc > 7 ? std::atoi(argv[7]) : 0;
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
    // DQ_HEAP (MB) enlarges the device malloc heap for kernels that allocate temporaries.
    if (auto heap = std::getenv("DQ_HEAP")) CHECK(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, size_t(std::atoi(heap)) << 20));
    if (smem) CHECK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));

    std::vector<float> a(N * N), s(N), c(N * N);
    std::vector<int8_t> w(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = float(i % 97) / 97.0f - 0.5f;
        w[i] = int8_t(int(i % 251) - 125);
    }
    for (size_t j = 0; j < N; ++j) s[j] = 0.01f + float(j % 13) / 1300.0f;
    std::array<size_t, 4> bytes{N * N * 4, N * N, N * 4, N * N * 4};
    std::array<const void*, 3> host{a.data(), w.data(), s.data()};
    std::array<CUdeviceptr, 4> dev_buf;
    // Slot `p` of the kernel's operand array holds operand `perm[p]` (0 = A, 1 = W, 2 = s, 3 = C).
    auto run = [&](const std::array<int, 4>& perm) {
        for (int p = 0; p != 4; ++p) {
            auto o = perm[p];
            CHECK(cuMemAlloc(&dev_buf[p], bytes[o]));
            if (o < 3) CHECK(cuMemcpyHtoD(dev_buf[p], host[o], bytes[o]));
            else CHECK(cuMemsetD8(dev_buf[p], 0, bytes[o]));
        }
    };
    auto check = [&](const std::array<int, 4>& perm) {
        run(perm);
        void* params[] = {dev_buf.data()};
        auto ok = cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, smem, nullptr, params, nullptr) == CUDA_SUCCESS
               && cuCtxSynchronize() == CUDA_SUCCESS;
        if (ok) {
            auto out = std::find(perm.begin(), perm.end(), 3) - perm.begin();
            CHECK(cuMemcpyDtoH(c.data(), dev_buf[out], bytes[3]));
            for (size_t t = 0; t < 64 && ok; ++t) {
                size_t i = (t * 131 + 7) % N, j = (t * 977 + 3) % N;
                double ref = 0;
                for (size_t k = 0; k < N; ++k) ref += double(a[i * N + k]) * double(w[k * N + j]) * double(s[j]);
                ok = std::fabs(ref - c[i * N + j]) <= 1e-2 * (1 + std::fabs(ref));
            }
        }
        if (!ok)
            for (auto p : dev_buf) cuMemFree(p);
        return ok;
    };
    // DQ_SIZES lists the byte size of each kernel slot, which rules out orders that would fault.
    std::array<size_t, 4> slot{};
    if (auto env = std::getenv("DQ_SIZES"))
        std::sscanf(env, "%zu,%zu,%zu,%zu", &slot[0], &slot[1], &slot[2], &slot[3]);
    auto fits = [&](const std::array<int, 4>& perm) {
        for (int p = 0; p != 4; ++p)
            if (slot[p] && slot[p] != bytes[perm[p]]) return false;
        return true;
    };
    std::array<int, 4> perm{0, 1, 2, 3};
    bool found = false;
    do found = fits(perm) && check(perm);
    while (!found && std::next_permutation(perm.begin(), perm.end()));
    if (!found) {
        std::printf("FAIL\n");
        return 1;
    }
    void* params[] = {dev_buf.data()};
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
    std::printf("OK median %.4f ms  min %.4f ms  %.0f GFLOP/s\n", med, ms[0], 2.0 * N * N * N / (med * 1e-3) / 1e9);
}
