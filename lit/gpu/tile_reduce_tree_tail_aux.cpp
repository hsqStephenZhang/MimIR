#include <cstdint>
#include <cstdio>

extern "C" std::int64_t* rowsum64(std::int64_t*);

int main() {
    std::int64_t x[6 * 64];
    for (int i = 0; i != 6 * 64; ++i)
        x[i] = i + 1;
    auto y       = rowsum64(x);
    int failures = 0;
    for (int i = 0; i != 6; ++i) {
        std::int64_t ref = 0;
        for (int j = 0; j != 64; ++j)
            ref += x[i * 64 + j];
        if (y[i] != ref) {
            std::printf("y[%d] = %ld, expected %ld\n", i, static_cast<long>(y[i]), static_cast<long>(ref));
            ++failures;
        }
    }
    return failures;
}
