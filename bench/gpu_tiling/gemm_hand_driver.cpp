// Times a hand-written `FN(a, b, c)` (C = A·B, 1024³, row-major F32) and checks sampled entries against a CPU reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef FN
#    define FN gemm_g1
#endif
#define STR_(x) #x
#define STR(x) STR_(x)

extern "C" int FN(float* a, float* b, float* c);

int main(int argc, char** argv) {
    constexpr size_t N = 1024;
    int iters          = argc > 1 ? std::atoi(argv[1]) : 10;
    std::vector<float> a(N * N), b(N * N), c(N * N, -1.0f);
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = float(i % 97) / 97.0f - 0.5f;
        b[i] = float(i % 89) / 89.0f - 0.5f;
    }
    FN(a.data(), b.data(), c.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) FN(a.data(), b.data(), c.data());
    auto t1   = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("%s end-to-end: %.3f ms/call\n", STR(FN), ms);
    int bad = 0;
    for (size_t s = 0; s < 256; ++s) {
        size_t i = (s * 131 + 7) % N, j = (s * 977 + 3) % N;
        double ref = 0;
        for (size_t k = 0; k < N; ++k) ref += double(a[i * N + k]) * double(b[k * N + j]);
        if (std::fabs(ref - c[i * N + j]) > 1e-2 * (1 + std::fabs(ref))) {
            if (bad < 5) std::printf("mismatch C[%zu][%zu]: got %g want %g\n", i, j, c[i * N + j], ref);
            ++bad;
        }
    }
    std::printf("%s\n", bad ? "FAIL" : "OK");
    return bad ? 1 : 0;
}
