// Runs a compiled 16-bit `FN(a, b)` (C = A·B, N³, row-major F16 inputs, F32 result; -DN, -DFN) and checks sampled entries.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef N
#define N 1024
#endif

extern "C" float* FN(_Float16* a, _Float16* b);

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 1;
    std::vector<_Float16> a(size_t(N) * N), b(size_t(N) * N);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = _Float16(float(i % 97) / 97.0f - 0.5f);
        b[i] = _Float16(float(i % 89) / 89.0f - 0.5f);
    }
    float* c = nullptr;
    for (int it = 0; it < iters; ++it) {
        if (c) std::free(c);
        c = FN(a.data(), b.data());
    }
    int bad = 0;
    for (size_t s = 0; s < 64; ++s) {
        size_t i = (s * 131 + 7) % N, j = (s * 977 + 3) % N;
        double ref = 0;
        for (size_t k = 0; k < N; ++k) ref += double(float(a[i * N + k])) * double(float(b[k * N + j]));
        if (std::fabs(ref - c[i * N + j]) > 2e-2 * (1 + std::fabs(ref))) {
            if (bad < 3) std::printf("C[%zu][%zu] = %g want %g\n", i, j, c[i * N + j], ref);
            ++bad;
        }
    }
    std::printf("%s\n", bad ? "FAIL" : "OK");
    return bad != 0;
}
