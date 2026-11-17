#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" float* gemm_refill(_Float16*, _Float16*);
extern "C" float* gemm_refill_late(_Float16*, _Float16*);

template<int K>
int check(float* (*gemm)(_Float16*, _Float16*)) {
    constexpr int m = 128, n = 128;
    static _Float16 a[m * K], b[n * K];
    for (int i = 0; i != m * K; ++i)
        a[i] = _Float16(float(i % 7) - 3.0f);
    for (int i = 0; i != n * K; ++i)
        b[i] = _Float16(float(i % 5) * 0.5f - 1.0f);
    auto c       = gemm(a, b);
    int failures = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float ref = 0.0f;
            for (int l = 0; l != K; ++l)
                ref += float(a[i * K + l]) * float(b[l * n + j]);
            if (std::fabs(ref - c[i * n + j]) > 1e-3f) {
                if (failures < 4) std::printf("K=%d c[%d][%d] = %g, expected %g\n", K, i, j, c[i * n + j], ref);
                ++failures;
            }
        }
    std::free(c);
    return failures;
}

int main() {
    int failures = check<256>(gemm_refill) + check<512>(gemm_refill_late);
    std::printf("failures=%d\n", failures);
    return failures ? 1 : 0;
}
