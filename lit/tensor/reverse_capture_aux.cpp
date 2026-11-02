#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <array>

extern "C" float* rows_17(float*, float*, float*);
extern "C" float* rows_33(float*, float*, float*);

static uint64_t additions;
extern "C" void record_add(uint64_t lanes) { additions += lanes; }

int main() {
    std::array<float, 33 * 8> x;
    std::array<float, 8> w, b;
    for (size_t j = 0; j != 8; ++j) {
        w[j] = float(j + 1) / 8;
        b[j] = float(j) / 16;
    }
    for (size_t i = 0; i != x.size(); ++i)
        x[i] = float(int(i * 7 % 23) - 11) / 4;
    for (auto rows : {17, 33}) {
        additions = 0;
        auto out  = (rows == 17 ? rows_17 : rows_33)(x.data(), w.data(), b.data());
        std::printf("rows=%d additions=%llu\n", rows, static_cast<unsigned long long>(additions));
        assert(additions <= uint64_t(4 * rows * 8 + 2 * rows));
        for (int i = 0; i != rows; ++i) {
            double mean = 0, variance = 0;
            for (int j = 0; j != 8; ++j)
                mean += x[i * 8 + j] / 8.0;
            for (int j = 0; j != 8; ++j)
                variance += std::pow(x[i * 8 + j] - mean, 2) / 8;
            for (int j = 0; j != 8; ++j) {
                auto expected = (x[i * 8 + j] - mean) / std::sqrt(variance + 0.00001) * w[j] + b[j];
                assert(std::abs(out[i * 8 + j] - expected) < 0.00001);
            }
        }
        std::free(out);
    }
}
