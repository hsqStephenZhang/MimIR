#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>

extern "C" {
float* nested_denote(float*, int8_t*, float*);
float* nested_rows(float*, int8_t*, float*);
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
    for (auto sums : {nested_denote, nested_rows}) {
        auto z = sums(a, w, s);
        for (int i = 0; i != m; ++i) {
            float ref = 0.0f;
            for (int j = 0; j != n; ++j)
                for (int l = 0; l != k; ++l)
                    ref += a[i * k + l] * (float(w[(l + shift) * n + j]) * s[j]);
            if (std::fabs(ref - z[i]) > 1e-3f * (1.0f + std::fabs(ref))) {
                std::printf("z[%d] = %g, expected %g\n", i, z[i], ref);
                ++failures;
            }
        }
        std::free(z);
    }
    return failures ? 1 : 0;
}
