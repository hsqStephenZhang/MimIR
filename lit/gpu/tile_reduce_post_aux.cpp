#include <cstdint>
#include <cstdio>

extern "C" std::int64_t* sum_count16(std::int64_t*);

int main() {
    constexpr int n = 16;
    std::int64_t x[n * n];
    for (int i = 0; i != n * n; ++i)
        x[i] = i % 13 - 6;
    auto y       = sum_count16(x);
    int failures = 0;
    for (int i = 0; i != n; ++i) {
        std::int64_t ref = n;
        for (int j = 0; j != n; ++j)
            ref += x[i * n + j];
        if (y[i] != ref) {
            std::printf("y[%d] = %ld, expected %ld\n", i, static_cast<long>(y[i]), static_cast<long>(ref));
            ++failures;
        }
    }
    return failures;
}
