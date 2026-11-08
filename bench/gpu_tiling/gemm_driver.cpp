// Times `gemm1024` (C = A·B, 1024³, row-major F32) and checks a sample of entries against a CPU reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" float* gemm1024(float* a, float* b);

int main(int argc, char** argv) {
    constexpr size_t N = 1024;
    int iters          = argc > 1 ? std::atoi(argv[1]) : 10;
    std::vector<float> a(N * N), b(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = float(i % 97) / 97.0f - 0.5f;
        b[i] = float(i % 89) / 89.0f - 0.5f;
    }
    float* c = gemm1024(a.data(), b.data());
    auto t0  = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) c = gemm1024(a.data(), b.data());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("gemm1024 end-to-end: %.3f ms/call  (%.1f GFLOP/s incl. copies)\n", ms, 2.0 * N * N * N / ms / 1e6);
    int bad = 0;
    for (size_t s = 0; s < 64; ++s) {
        size_t i = (s * 131) % N, j = (s * 977) % N;
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
