#include <cstdio>
extern "C" int cluster_sum(float* x, float* out);
int main() {
    static float x[2 * 128 * 8], out[128];
    for (int i = 0; i < 2 * 128 * 8; ++i) x[i] = float(i % 13);
    cluster_sum(x, out);
    int bad = 0;
    for (int t = 0; t < 128; ++t) {
        float want = 0;
        for (int g = 0; g < 2; ++g)
            for (int j = 0; j < 8; ++j) want += x[(g * 128 + t) * 8 + j];
        if (out[t] != want) {
            if (bad < 4) std::printf("out[%d] = %g, want %g\n", t, out[t], want);
            ++bad;
        }
    }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
