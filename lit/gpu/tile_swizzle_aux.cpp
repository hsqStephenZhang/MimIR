#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" float* gemm64(float*, float*);

int main() {
    constexpr int n = 64;
    float a[n * n], b[n * n];
    for (int i = 0; i != n * n; ++i) {
        a[i] = float(i % 7) - 3.0f;
        b[i] = float(i % 5) * 0.5f - 1.0f;
    }
    auto c       = gemm64(a, b);
    int failures = 0;
    for (int i = 0; i != n; ++i)
        for (int j = 0; j != n; ++j) {
            float ref = 0.0f;
            for (int k = 0; k != n; ++k)
                ref += a[i * n + k] * b[k * n + j];
            if (std::fabs(ref - c[i * n + j]) > 1e-2f) {
                std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
                ++failures;
            }
        }
    std::free(c);
    return failures ? 1 : 0;
}
