#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef K
#    define K 128
#endif

extern "C" _Float16* gemm_wg_h(_Float16*, _Float16*);

int main() {
    constexpr int m = 128, n = 128, k = K;
    static _Float16 a[m * k], b[n * k];
    for (int i = 0; i != m * k; ++i)
        a[i] = _Float16(float(i % 7) - 3.0f);
    for (int i = 0; i != n * k; ++i)
        b[i] = _Float16(float(i % 5) * 0.5f - 1.0f);
    auto c       = gemm_wg_h(a, b);
    int failures = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float ref = 0.0f;
            for (int l = 0; l != k; ++l)
                ref += float(a[i * k + l]) * float(b[l * n + j]);
            float got = float(c[i * n + j]);
            if (std::fabs(ref - got) > 1e-2f * (1.0f + std::fabs(ref))) {
                if (failures < 4) std::printf("c[%d][%d] = %g, expected %g\n", i, j, got, ref);
                ++failures;
            }
        }
    std::free(c);
    std::printf("failures=%d\n", failures);
    return failures ? 1 : 0;
}
