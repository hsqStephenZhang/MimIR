#include <cstdio>
extern "C" int tma_box(_Float16* a, _Float16* out);
int main() {
    static _Float16 a[128 * 64], out[64 * 64];
    for (int i = 0; i < 128 * 64; ++i) a[i] = _Float16(float(i % 1021));
    tma_box(a, out);
    int bad = 0;
    for (int i = 0; i < 64 * 64; ++i)
        if (out[i] != a[64 * 64 + i]) {
            if (bad < 4) std::printf("out[%d][%d] = %g, want %g\n", i / 64, i % 64, float(out[i]), float(a[64 * 64 + i]));
            ++bad;
        }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
