#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

extern "C" {
float* fuse_denote(float*, int8_t*, float*);
float* fuse_staged(float*, int8_t*, float*);
float* fuse_unstaged(float*, int8_t*, float*);
float* fuse_req(float*, int8_t*, float*);
}

int main() {
    constexpr int m = 32, k = 16, n = 32, rows = 21, shift = 5;
    float a[m * k], s[n];
    int8_t w[rows * n];
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != rows * n; ++i)
        w[i] = int8_t(i % 23 - 11);
    for (int j = 0; j != n; ++j)
        s[j] = 0.25f + float(j % 3) * 0.125f;
    int failures = 0;
    for (auto gemm : {fuse_denote, fuse_staged, fuse_unstaged, fuse_req}) {
        auto c = gemm(a, w, s);
        for (int i = 0; i != m; ++i)
            for (int j = 0; j != n; ++j) {
                float ref = 0.0f;
                for (int l = 0; l != k; ++l)
                    ref += a[i * k + l] * (float(w[(l + shift) * n + j]) * s[j]);
                if (std::fabs(ref - c[i * n + j]) > 1e-3f * (1.0f + std::fabs(ref))) {
                    std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
                    ++failures;
                }
            }
        std::free(c);
    }
    return failures ? 1 : 0;
}
