#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" float* soft(float*);
int main() {
    int m = 5, n = 7, bad = 0;
    float x[5 * 7];
    for (int i = 0; i != m * n; ++i) x[i] = (i * 7 % 13) / 3.f - 2.f;
    for (int rep = 0; rep != 3; ++rep) {
        float* r = soft(x);
        for (int i = 0; i != m; ++i) {
            double top = x[i * n], sum = 0;
            for (int j = 0; j != n; ++j) top = std::fmax(top, x[i * n + j]);
            for (int j = 0; j != n; ++j) sum += std::exp(x[i * n + j] - top);
            for (int j = 0; j != n; ++j)
                bad += std::fabs(std::exp(x[i * n + j] - top) / sum - r[i * n + j]) > 1e-5;
        }
        free(r);
    }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
