#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* mm49(float*, float*);
float* mm65(float*, float*);
float* mm130(float*, float*);
}

static int check(float* (*product)(float*, float*), size_t M, size_t K, size_t N) {
    std::vector<float> a(M * K), b(K * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = float(i % 13) / 13.0f - 0.4f;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = float((7 * i) % 11) / 11.0f - 0.5f;
    float* out = product(a.data(), b.data());
    int bad    = 0;
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            double expected = 0;
            for (size_t k = 0; k < K; ++k)
                expected += double(a[m * K + k]) * b[k * N + n];
            if (!std::isfinite(out[m * N + n]) || std::fabs(out[m * N + n] - expected) > 1e-3) ++bad;
        }
    std::free(out);
    if (bad) std::fprintf(stderr, "%zu x %zu x %zu: %d incorrect elements\n", M, K, N, bad);
    return bad;
}

int main() { return check(mm49, 64, 49, 1024) + check(mm65, 32, 65, 512) + check(mm130, 64, 130, 512) ? 1 : 0; }
