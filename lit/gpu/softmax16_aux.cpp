#include <cmath>
#include <cstdio>

extern "C" float* softmax16(float*);

int main() {
    constexpr int n = 16;
    float x[n * n];
    for (int i = 0; i != n * n; ++i)
        x[i] = float((i * 7) % 11) * 0.25f - 1.0f;
    auto y       = softmax16(x);
    int failures = 0;
    for (int i = 0; i != n; ++i) {
        float m = -1e30f;
        for (int j = 0; j != n; ++j)
            m = std::fmax(m, x[i * n + j]);
        double l = 0;
        for (int j = 0; j != n; ++j)
            l += std::exp(double(x[i * n + j] - m));
        for (int j = 0; j != n; ++j) {
            auto expected = std::exp(double(x[i * n + j] - m)) / l;
            if (std::fabs(expected - y[i * n + j]) > 1e-5 * expected + 1e-7) {
                std::printf("(%d, %d): %g, expected %g\n", i, j, y[i * n + j], expected);
                ++failures;
            }
        }
    }
    return failures;
}
