// kbench16 <cubin> <kernel> <groups> <items> <f16|bf16> [iters] [n] [smem]: times a generated n³ 16-bit GEMM kernel back to back.
// The kernel takes its three device buffers as one by-value array (A, B, C), row-major, C in the input type.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template<class H> int run(int argc, char** argv) {
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

    std::vector<H> a(N * N), b(N * N), c(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = H(float(i % 97) / 97.0f - 0.5f);
        b[i] = H(float(i % 89) / 89.0f - 0.5f);
    }
    std::array<CUdeviceptr, 3> buf;
    for (auto& p : buf) CHECK(cuMemAlloc(&p, N * N * sizeof(H)));
    CHECK(cuMemcpyHtoD(buf[0], a.data(), N * N * sizeof(H)));
    CHECK(cuMemcpyHtoD(buf[1], b.data(), N * N * sizeof(H)));
    CHECK(cuMemsetD8(buf[2], 0, N * N * sizeof(H)));
    void* params[] = {buf.data()};
    CHECK(cuLaunchKernel(fn, groups, 1, 1, items, 1, 1, smem, nullptr, params, nullptr));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(c.data(), buf[2], N * N * sizeof(H)));
    int bad = 0;
    for (size_t s = 0; s < 256; ++s) {
        size_t i = (s * 131 + 7) % N, j = (s * 977 + 3) % N;
        double ref = 0;
        for (size_t k = 0; k < N; ++k) ref += double(float(a[i * N + k])) * double(float(b[k * N + j]));
        double got = float(c[i * N + j]);
        if (std::fabs(ref - got) > 2e-2 * (1 + std::fabs(ref))) {
            if (bad < 3) std::printf("C[%zu][%zu] = %g want %g\n", i, j, got, ref);
            ++bad;
        }
    }
    if (bad) {
        std::printf("FAIL (%d)\n", bad);
        return 1;
    }
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
    return std::strcmp(argv[5], "bf16") == 0 ? run<Bf16>(argc, argv) : run<_Float16>(argc, argv);
}
