#include <cassert>
#include <cmath>
#include <cstdlib>

#include <array>

extern "C" float* dot(float*, float*);

int main() {
    std::array<float, 256 * 4> a, b;
    for (size_t i = 0; i != a.size(); ++i) {
        a[i] = float(int(i % 17) - 8) / 8;
        b[i] = float(int(i % 13) - 6) / 8;
    }
    auto out = dot(a.data(), b.data());
    for (size_t h = 0; h != 256; ++h)
        for (size_t i = 0; i != 2; ++i)
            for (size_t j = 0; j != 2; ++j) {
                double expected = 0;
                for (size_t k = 0; k != 2; ++k)
                    expected += double(a[h * 4 + i * 2 + k]) * b[h * 4 + k * 2 + j];
                assert(std::abs(out[h * 4 + i * 2 + j] - expected) < 0.00001);
            }
    std::free(out);
}
