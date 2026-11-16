#include <cmath>
#include <cstdio>
extern "C" float* rowsum16(float*);
int main() {
    float x[256];
    for (int i = 0; i != 256; ++i) x[i] = float(i % 11) - 4.0f + 0.25f * float(i % 3);
    auto got = rowsum16(x);
    int bad = 0;
    for (int i = 0; i != 16; ++i) { float w = 0; for (int j = 0; j != 16; ++j) w += x[i * 16 + j]; if (std::fabs(got[i] - w) > 1e-3f) ++bad; }
    std::printf("%d wrong of 16\n", bad);
    return bad != 0;
}
