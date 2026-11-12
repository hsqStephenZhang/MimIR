#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" float* gemm(float*, int8_t*, float*);

int main() {
    constexpr int m = 32, n = 32, k = 16, rows = 21;
    float a[m * k], s[n];
    int8_t w[rows * n];
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i * 7 % 11) - 5.0f;
    for (int i = 0; i != rows * n; ++i)
        w[i] = int8_t(i * 13 % 23 - 11);
    for (int j = 0; j != n; ++j)
        s[j] = 0.25f + float(j % 3) * 0.125f;
    auto c       = gemm(a, w, s);
    int failures = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float ref = 0.0f;
            for (int l = 0; l != k; ++l)
                ref += a[i * k + l] * (float(w[(l + 5) * n + j]) * s[j]);
            if (std::fabs(ref - c[i * n + j]) > 1e-2f * (1.0f + std::fabs(ref))) {
                std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
                ++failures;
            }
        }
    std::free(c);
    return failures ? 1 : 0;
}
