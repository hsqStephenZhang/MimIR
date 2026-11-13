#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" float* gemm32(_Float16*, _Float16*);

int main() {
    constexpr int m = 32, k = 64;
    static _Float16 a[m * k], b[m * k];
    for (int i = 0; i != m * k; ++i) {
        a[i] = _Float16(float(i % 7) - 3.0f);
        b[i] = _Float16(float(i % 5) * 0.5f - 1.0f);
    }
    auto c       = gemm32(a, b);
    int failures = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != m; ++j) {
            float ref = 0.0f;
            for (int l = 0; l != k; ++l) ref += float(a[i * k + l]) * float(b[j * k + l]);
            if (std::fabs(ref - c[i * m + j]) > 1e-3f) {
                if (failures < 4) std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * m + j], ref);
                ++failures;
            }
        }
    std::free(c);
    std::printf("failures=%d\n", failures);
    return failures ? 1 : 0;
}
