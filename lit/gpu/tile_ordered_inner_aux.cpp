#include <cstdint>
#include <cstdio>

extern "C" std::int64_t* ordered_i64(std::int64_t*);

int main() {
    std::int64_t x[2 * 4 * 6];
    for (int i = 0; i != 2 * 4 * 6; ++i)
        x[i] = i % 9 - 4;
    auto y       = ordered_i64(x);
    int failures = 0;
    for (int i = 0; i != 2; ++i) {
        std::int64_t ref = 0;
        for (int j = 0; j != 4 * 6; ++j)
            ref += x[i * 4 * 6 + j];
        if (y[i] != ref) {
            std::printf("y[%d] = %ld, expected %ld\n", i, static_cast<long>(y[i]), static_cast<long>(ref));
            ++failures;
        }
    }
    return failures;
}
