#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" float* gemm_tail(float*, float*);

int main() {
    constexpr int m = 32, k = 24, n = 32;
    float a[m * k], b[k * n];
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != k * n; ++i)
        b[i] = float(i % 5) * 0.5f - 1.0f;
    auto c       = gemm_tail(a, b);
    int failures = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float ref = 0.0f;
            for (int l = 0; l != k; ++l)
                ref += a[i * k + l] * b[l * n + j];
            if (std::fabs(ref - c[i * n + j]) > 1e-3f) {
                std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
                ++failures;
            }
        }
    std::free(c);
    return failures ? 1 : 0;
}
