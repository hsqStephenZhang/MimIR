#include <cmath>
#include <cstdio>

extern "C" float* rowsum16(float*);

int main() {
    constexpr int n = 16;
    float x[n * n];
    for (int i = 0; i != n * n; ++i) x[i] = float(i % 7) - 2.5f;
    auto y       = rowsum16(x);
    int failures = 0;
    for (int i = 0; i != n; ++i) {
        float ref = 0.0f;
        for (int j = 0; j != n; ++j) ref += x[i * n + j];
        if (std::fabs(ref - y[i]) > 1e-4f) {
            std::printf("y[%d] = %g, expected %g\n", i, y[i], ref);
            ++failures;
        }
    }
    return failures;
}
