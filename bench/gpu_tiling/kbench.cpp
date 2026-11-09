// kbench <cubin> <kernel> <groups> <items> [iters] [n]: times a generated n³ SGEMM kernel back to back in one context.
// The kernel takes its three device buffers as one by-value array; the operand order is found by checking the result.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda.h>

#define CHECK(x)                                                                         \
    do {                                                                                 \
        if (auto r = (x); r != CUDA_SUCCESS) {                                           \
            std::fprintf(stderr, "%s failed: %d\n", #x, int(r));                          \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)


int main(int argc, char** argv) {
    int groups = std::atoi(argv[3]), items = std::atoi(argv[4]), iters = argc > 5 ? std::atoi(argv[5]) : 300;
    size_t N   = argc > 6 ? std::atoi(argv[6]) : 1024;
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

    std::vector<float> a(N * N), b(N * N), c(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = float(i % 97) / 97.0f - 0.5f;
        b[i] = float(i % 89) / 89.0f - 0.5f;
    }
    std::array<CUdeviceptr, 3> buf;
    for (auto& p : buf) CHECK(cuMemAlloc(&p, N * N * sizeof(float)));
    auto check = [&](int ia, int ib, int ic) {
        CHECK(cuMemcpyHtoD(buf[ia], a.data(), N * N * 4));
        CHECK(cuMemcpyHtoD(buf[ib], b.data(), N * N * 4));
        CHECK(cuMemsetD32(buf[ic], 0, N * N));
        void* params[] = {buf.data()};
        CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, 0, nullptr, params, nullptr));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(c.data(), buf[ic], N * N * 4));
        for (size_t s = 0; s < 64; ++s) {
            size_t i = (s * 131 + 7) % N, j = (s * 977 + 3) % N;
            double ref = 0;
            for (size_t k = 0; k < N; ++k) ref += double(a[i * N + k]) * double(b[k * N + j]);
            if (std::fabs(ref - c[i * N + j]) > 1e-2 * (1 + std::fabs(ref))) return false;
        }
        return true;
    };
    int order[3] = {-1, -1, -1};
    for (int ia = 0; ia < 3 && order[0] < 0; ++ia)
        for (int ib = 0; ib < 3; ++ib)
            if (ib != ia && check(ia, ib, 3 - ia - ib)) {
                order[0] = ia, order[1] = ib, order[2] = 3 - ia - ib;
                break;
            }
    if (order[0] < 0) {
        std::printf("FAIL\n");
        return 1;
    }
    void* params[] = {buf.data()};
    for (int i = 0; i < 20; ++i) CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, 0, nullptr, params, nullptr));
    CUevent start, stop;
    CHECK(cuEventCreate(&start, 0));
    CHECK(cuEventCreate(&stop, 0));
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i) {
        CHECK(cuEventRecord(start, nullptr));
        CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, 0, nullptr, params, nullptr));
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
