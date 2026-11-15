#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

extern "C" {
float* mma_run(float*, int8_t*, float*);
float* mma_req(float*, int8_t*, float*);
float* simt_run(float*, int8_t*, float*);
float* simt_req(float*, int8_t*, float*);
float* item_run(float*, int8_t*, float*);
}

int main() {
    constexpr int n = 16;
    float a[n * n], s[n];
    int8_t w[n * n];
    for (int i = 0; i != n * n; ++i) {
        a[i] = float(i % 7) - 3.0f;
        w[i] = int8_t(i % 23 - 11);
    }
    for (int j = 0; j != n; ++j)
        s[j] = 0.25f + float(j % 3) * 0.125f;
    int failures = 0;
    for (auto gemm : {mma_run, mma_req, simt_run, simt_req, item_run}) {
        auto c = gemm(a, w, s);
        for (int i = 0; i != n; ++i)
            for (int j = 0; j != n; ++j) {
                float ref = 0.0f;
                for (int k = 0; k != n; ++k)
                    ref += a[i * n + k] * (float(w[k * n + j]) * s[j]);
                if (std::fabs(ref - c[i * n + j]) > 1e-2f * (1.0f + std::fabs(ref))) {
                    std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
                    ++failures;
                }
            }
        std::free(c);
    }
    return failures ? 1 : 0;
}
