#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" float* cross_entropy(float*, int32_t*);

int main() {
    constexpr int m = 8, v = 4096;
    static float x[m * v];
    static int32_t t[m];
    for (int i = 0; i != m * v; ++i) x[i] = float((i * 37) % 101) / 25.0f - 2.0f;
    for (int i = 0; i != m; ++i) t[i] = (i * 977 + 13) % v;
    auto y       = cross_entropy(x, t);
    int failures = 0;
    for (int i = 0; i != m; ++i) {
        double mx = -1e30, sum = 0;
        for (int j = 0; j != v; ++j) mx = std::fmax(mx, x[i * v + j]);
        for (int j = 0; j != v; ++j) sum += std::exp(double(x[i * v + j]) - mx);
        double ref = mx + std::log(sum) - x[i * v + t[i]];
        if (std::fabs(ref - y[i]) > 1e-3 * (1 + std::fabs(ref))) {
            std::printf("y[%d] = %g, expected %g\n", i, y[i], ref);
            ++failures;
        }
    }
    std::free(y);
    std::printf("failures=%d\n", failures);
    return failures ? 1 : 0;
}
