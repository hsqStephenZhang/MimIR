#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" float* gemm16(float*, float*);
int main() {
    constexpr int n = 16;
    float a[n * n], b[n * n];
    for (int i = 0; i != n * n; ++i) { a[i] = float(i % 7) - 3.0f; b[i] = float(i % 5) * 0.5f - 1.0f; }
    auto got = gemm16(a, b);
    int bad = 0;
    for (int i = 0; i != n; ++i)
        for (int j = 0; j != n; ++j) {
            float w = 0;
            for (int l = 0; l != n; ++l) w += a[i * n + l] * b[l * n + j];
            if (std::fabs(got[i * n + j] - w) > 1e-3f) ++bad;
        }
    std::printf("%d wrong of %d\n", bad, n * n);
    return bad != 0;
}
