// Times one GEMM C = A·B compiled by two mim builds (`gemm_new`, `gemm_base`) after checking them against each other.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* gemm_new(float*, float*);
float* gemm_base(float*, float*);
}

int main(int, char** argv) {
    std::vector<float> a(size_t(M) * K), b(size_t(K) * N);
    for (size_t i = 0; i != a.size(); ++i) a[i] = float(int(i * 7 % 11) - 5) / 8;
    for (size_t i = 0; i != b.size(); ++i) b[i] = float(int(i * 5 % 13) - 6) / 32;
    float *x = gemm_new(a.data(), b.data()), *y = gemm_base(a.data(), b.data());
    int bad = 0;
    for (size_t i = 0; i != size_t(M) * N; ++i)
        if (std::fabs(x[i] - y[i]) > 1e-3f * (1 + std::fabs(y[i]))) ++bad;
    std::free(x), std::free(y);
    double flop = 2.0 * M * N * K;
    auto time = [&](float* (*f)(float*, float*)) {
        double best = 1e30;
        int reps    = std::max(20, int(2e9 / flop));
        for (int r = 0; r != reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            std::free(f(a.data(), b.data()));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return flop / best / 1e9;
    };
    double n0 = time(gemm_new), b0 = time(gemm_base), n1 = time(gemm_new), b1 = time(gemm_base);
    double n = std::max(n0, n1), bs = std::max(b0, b1);
    std::printf("%-14s wrong=%d  alloc %7.1f  base %7.1f GFLOP/s  alloc/base %.3f\n", argv[1], bad, n, bs, n / bs);
    return bad != 0;
}
