#include <cmath>
#include <cstdio>
extern "C" float* gemm(float*, float*, float*);
int main() {
    static float a[64 * 64], b[64 * 64], c0[64 * 64];
    for (int i = 0; i < 64 * 64; ++i)
        a[i] = float(i % 7) - 3, b[i] = float(i % 5) * 0.5f - 1, c0[i] = float(i % 11) - 5;
    float* c = gemm(a, b, c0);
    int bad  = 0;
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j) {
            float r = c0[i * 64 + j];
            for (int k = 0; k < 64; ++k)
                r += a[i * 64 + k] * b[k * 64 + j];
            if (std::fabs(r - c[i * 64 + j]) > 1e-3f) {
                if (bad < 3) std::printf("c[%d][%d]=%g want %g\n", i, j, c[i * 64 + j], r);
                ++bad;
            }
        }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
