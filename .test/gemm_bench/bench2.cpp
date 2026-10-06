#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using F = float* (*)(float*, float*);
extern "C" {
#define S(n, M, K, N, R) float* n(float*, float*);
#include "shapes.inc"
#undef S
}
struct Shape { const char* name; F f; size_t M, K, N; int reps; };
static Shape shapes[] = {
#define S(n, M, K, N, R) {#n, n, M, K, N, R},
#include "shapes.inc"
#undef S
};
using clk = std::chrono::steady_clock;
template<class G> static double best_secs(G&& g, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now(); g(); auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}
int main(int argc, char** argv) {
    auto wanted = [&](const char* n) { if (argc < 2) return true; for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], n)) return true; return false; };
    std::printf("%-9s %5s %5s %5s | %9s %9s | %9s %9s | %7s | %s\n", "name", "M", "K", "N", "mim_ms", "mim_GF", "blas_ms", "blas_GF", "ratio", "maxerr");
    for (auto& s : shapes) {
        if (!wanted(s.name)) continue;
        size_t M = s.M, K = s.K, N = s.N;
        std::vector<float> a(M * K), b(K * N), c(M * N);
        for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i % 13) / 13.0f - 0.4f;
        for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>((7 * i) % 11) / 11.0f - 0.5f;
        double flop = 2.0 * M * N * K;
        auto blas = [&] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a.data(), K, b.data(), N, 0.0f, c.data(), N); };
        float* out = s.f(a.data(), b.data());
        blas();
        double err = 0;
        for (size_t i = 0; i < c.size(); ++i) err = std::max(err, (double)std::fabs(out[i] - c[i]));
        free(out);
        double t_mim = best_secs([&] { float* o = s.f(a.data(), b.data()); free(o); }, s.reps);
        double t_b   = best_secs(blas, s.reps);
        std::printf("%-9s %5zu %5zu %5zu | %9.3f %9.2f | %9.3f %9.2f | %6.1f%% | %.1e\n", s.name, M, K, N, t_mim * 1e3,
                    flop / t_mim * 1e-9, t_b * 1e3, flop / t_b * 1e-9, 100.0 * t_b / t_mim, err);
        std::fflush(stdout);
    }
}
