#include <cmath>
#include <cstdint>
#include <cstdio>

extern "C" float* f32_fast(float*);
extern "C" std::int64_t* i64_add(std::int64_t*);

int main() {
    int failures = 0;
    float floats[16 * 16];
    for (int i = 0; i != 16 * 16; ++i)
        floats[i] = float(i % 11 - 5) * 0.25f;
    auto floats_y = f32_fast(floats);
    for (int i = 0; i != 16; ++i) {
        float ref = 0.0f;
        for (int j = 0; j != 16; ++j)
            ref += floats[i * 16 + j];
        if (std::fabs(floats_y[i] - ref) > 1e-5f) {
            std::printf("f32[%d] = %g, expected %g\n", i, floats_y[i], ref);
            ++failures;
        }
    }

    std::int64_t integers[20 * 20];
    for (int i = 0; i != 20 * 20; ++i)
        integers[i] = i % 17 - 8;
    auto integers_y = i64_add(integers);
    for (int i = 0; i != 20; ++i) {
        std::int64_t ref = 0;
        for (int j = 0; j != 20; ++j)
            ref += integers[i * 20 + j];
        if (integers_y[i] != ref) {
            std::printf("i64[%d] = %ld, expected %ld\n", i, static_cast<long>(integers_y[i]), static_cast<long>(ref));
            ++failures;
        }
    }
    return failures;
}
