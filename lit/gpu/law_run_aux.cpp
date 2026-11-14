#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>

extern "C" float* law_plain(float*, float*);
extern "C" float* law_fused(float*, float*);

int main() {
    constexpr int m = 16, n = 16, k = 20;
    float a[m * k], b[k * n], want[m * n];
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != k * n; ++i)
        b[i] = float(i % 5) * 0.5f - 1.0f;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            want[i * n + j] = 0.0f;
            for (int l = 0; l != k; ++l)
                want[i * n + j] += a[i * k + l] * b[l * n + j];
        }
    int failures = 0;
    for (auto f : {law_plain, law_fused}) {
        auto got = f(a, b);
        for (int i = 0; i != m * n; ++i)
            if (std::fabs(got[i] - want[i]) > 1e-3f) {
                std::printf("element %d is %f, not %f\n", i, got[i], want[i]);
                ++failures;
                break;
            }
        std::free(got);
    }
    return failures != 0;
}
