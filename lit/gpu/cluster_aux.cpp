#include <cstdio>
extern "C" int cluster_box(_Float16* a, _Float16* out);
int main() {
    static _Float16 a[64 * 64], out[2 * 64 * 64];
    for (int i = 0; i < 64 * 64; ++i) a[i] = _Float16(float(i % 1021));
    cluster_box(a, out);
    int bad = 0;
    for (int g = 0; g < 2; ++g)
        for (int i = 0; i < 64 * 64; ++i)
            if (out[g * 64 * 64 + i] != a[i]) {
                if (bad < 4) std::printf("out[%d][%d][%d] = %g, want %g\n", g, i / 64, i % 64, float(out[g * 64 * 64 + i]), float(a[i]));
                ++bad;
            }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
