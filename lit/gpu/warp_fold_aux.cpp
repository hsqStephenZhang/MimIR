#include <cmath>
#include <cstdio>
extern "C" float* warp_fold(float*);
int main() {
    static float x[256 * 64];
    for (int i = 0; i < 256 * 64; ++i) x[i] = float(i % 7) - 3;
    float* y = warp_fold(x);
    int bad = 0;
    // reference: xt for warp row block starting at row0 (global row of block start) and step ko=b*16: rows row0*4 + b*16 + i
    for (int r0 = 0; r0 < 64; r0 += 16) for (int i = 0; i < 16; ++i) {
        float l = 0; float o[64] = {0};
        for (int b = 0; b < 4; ++b) {
            int row = r0 * 4 + b * 16 + i;
            float rs = 0;
            for (int c = 0; c < 64; ++c) { o[c] = o[c] * 0.5f + x[row * 64 + c] + l; rs += x[row * 64 + c]; }
            l += rs;
        }
        for (int c = 0; c < 64; ++c) {
            float want = o[c] + l, got = y[(r0 + i) * 64 + c];
            if (std::fabs(got - want) > 1e-3f) { if (bad < 4) std::printf("y[%d][%d]=%g want %g\n", r0 + i, c, got, want); ++bad; }
        }
    }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
