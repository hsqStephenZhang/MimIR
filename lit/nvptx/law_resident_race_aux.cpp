#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" float* rows(float*);

int main() {
    constexpr int nrows = 64, cols = 256;
    static float x[nrows * cols];
    for (int i = 0; i != nrows * cols; ++i)
        x[i] = float(i % 113) / 37.0f - 1.5f;
    int failures = 0;
    for (int run = 0; run != 20 && !failures; ++run) {
        auto y = rows(x);
        for (int r = 0; r != nrows; ++r) {
            float mx = -1e30f;
            for (int c = 0; c != cols; ++c)
                mx = std::fmax(mx, x[r * cols + c]);
            double sum = 0;
            for (int c = 0; c != cols; ++c)
                sum += std::exp(double(x[r * cols + c] - mx));
            for (int c = 0; c != cols; ++c) {
                double ref = std::exp(double(x[r * cols + c] - mx)) / sum;
                if (std::fabs(ref - y[r * cols + c]) > 1e-4 * (1 + std::fabs(ref)) && failures++ < 5)
                    std::printf("run %d: y[%d][%d] = %g, expected %g\n", run, r, c, y[r * cols + c], ref);
            }
        }
        std::free(y);
    }
    return failures ? 1 : 0;
}
