#include <cmath>
#include <cstdio>
extern "C" float* gemm(float*, float*);
int main() {
    static float x[64 * 32], w[32 * 64];
    for (int i = 0; i < 64 * 32; ++i) x[i] = float(i % 7) - 3, w[i] = float(i % 5) - 2;
    float* y = gemm(x, w);
    int bad  = 0;
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j) {
            float s = 0;
            for (int k = 0; k < 32; ++k) s += x[i * 32 + k] * w[k * 64 + j];
            float want = 2 * s + 1;
            if (std::fabs(y[i * 64 + j] - want) > 1e-3f) {
                if (bad < 3) std::printf("y[%d][%d]=%g want %g\n", i, j, y[i * 64 + j], want);
                ++bad;
            }
        }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
