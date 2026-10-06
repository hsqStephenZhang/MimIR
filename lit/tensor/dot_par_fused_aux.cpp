#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* packed(float*, float*);
float* rectangular(float*, float*);
float* unpacked(float*, float*);
}

static bool check(float* (*f)(float*, float*), size_t M, size_t K, size_t N) {
    std::vector<float> a(M * K), b(K * N), expected(M * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = float(i % 13) / 13 - 0.4f;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = float((7 * i) % 11) / 11 - 0.5f;
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            double value = 0;
            for (size_t k = 0; k < K; ++k)
                value += double(a[m * K + k]) * b[k * N + n];
            expected[m * N + n] = float(value);
        }
    // Reuse the input while each invocation owns and releases a fresh packed operand.
    for (int rep = 0; rep < 5; ++rep) {
        auto out = f(a.data(), b.data());
        for (size_t i = 0; i < expected.size(); ++i)
            if (!std::isfinite(out[i]) || std::fabs(out[i] - expected[i]) > 1e-3) {
                std::fprintf(stderr, "%zux%zux%zu output[%zu] = %g, expected %g\n", M, K, N, i, out[i], expected[i]);
                std::free(out);
                return false;
            }
        std::free(out);
    }
    return true;
}

int main() {
    return check(packed, 256, 256, 256) && check(rectangular, 384, 256, 384) && check(unpacked, 32, 256, 512) ? 0 : 1;
}
