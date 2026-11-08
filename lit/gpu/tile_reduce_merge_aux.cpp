#include <cmath>
#include <cstdio>

extern "C" float* online16(float*);

int main() {
    constexpr int n = 16;
    float x[n * n];
    for (int i = 0; i != n * n; ++i) x[i] = float((i * 7) % 11) * 0.25f - 1.0f;
    auto y       = online16(x);
    int failures = 0;
    for (int i = 0; i != n; ++i) {
        float m = -1e30f;
        for (int j = 0; j != n; ++j) m = std::fmax(m, x[i * n + j]);
        double l = 0;
        for (int j = 0; j != n; ++j) l += std::exp(double(x[i * n + j] - m));
        if (std::fabs(m - y[2 * i]) > 1e-5f || std::fabs(l - y[2 * i + 1]) > 1e-4 * l) {
            std::printf("row %d: (%g, %g), expected (%g, %g)\n", i, y[2 * i], y[2 * i + 1], m, l);
            ++failures;
        }
    }
    return failures;
}
