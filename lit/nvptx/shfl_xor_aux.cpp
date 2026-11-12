#include <cstdio>

extern "C" int quads(float*);

int main() {
    float out[32];
    quads(out);
    int failures = 0;
    for (int lane = 0; lane != 32; ++lane) {
        float quad = float(4 * (lane / 4) * 4 + 6);
        if (out[lane] != quad) {
            std::printf("lane %d: %g, expected %g\n", lane, out[lane], quad);
            ++failures;
        }
    }
    return failures ? 1 : 0;
}
