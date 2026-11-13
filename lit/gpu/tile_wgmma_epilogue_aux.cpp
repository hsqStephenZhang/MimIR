#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" _Float16* gemm_bias(_Float16*, _Float16*, _Float16*);

int main() {
    constexpr int m = 200, n = 128, k = 128;
    static _Float16 a[m * k], b[k * n], bias[n];
    for (int i = 0; i != m * k; ++i) a[i] = _Float16(float(i % 7) - 3.0f);
    for (int i = 0; i != k * n; ++i) b[i] = _Float16(float(i % 5) * 0.5f - 1.0f);
    for (int i = 0; i != n; ++i) bias[i] = _Float16(float(i % 3) - 1.0f);
    auto c       = gemm_bias(a, b, bias);
    int failures = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float ref = float(bias[j]);
            for (int l = 0; l != k; ++l) ref += float(a[i * k + l]) * float(b[l * n + j]);
            if (std::fabs(ref - float(c[i * n + j])) > 1e-2f * (1 + std::fabs(ref))) {
                if (failures < 4) std::printf("c[%d][%d] = %g, expected %g\n", i, j, float(c[i * n + j]), ref);
                ++failures;
            }
        }
    std::free(c);
    std::printf("failures=%d\n", failures);
    return failures ? 1 : 0;
}
