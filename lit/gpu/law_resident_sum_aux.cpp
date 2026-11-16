#include <cmath>
#include <cstdio>

extern "C" float* rows(float*);

int main() {
    constexpr int n = 16;
    float x[n * n];
    for (int i = 0; i != n * n; ++i)
        x[i] = float((i * 7) % 11) * 0.25f + 0.5f;
    auto y       = rows(x);
    int failures = 0;
    for (int i = 0; i != n; ++i) {
        double s = 0;
        for (int j = 0; j != n; ++j)
            s += x[i * n + j];
        for (int j = 0; j != n; ++j) {
            auto expected = x[i * n + j] / s;
            if (std::fabs(expected - y[i * n + j]) > 1e-5 * expected + 1e-7) {
                std::printf("(%d, %d): %g, expected %g\n", i, j, y[i * n + j], expected);
                ++failures;
            }
        }
    }
    return failures;
}
