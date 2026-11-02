#include <cassert>
#include <cmath>
#include <cstdlib>

#include <array>

extern "C" float* batched(float*, float*, float*);
extern "C" float* batch_nt(float*, float*, float*);
extern "C" float* batch_tn(float*, float*, float*);
extern "C" float* single(float*, float*, float*);
extern "C" float* prefix(float*, float*, float*);

int main() {
    std::array<float, 6 * 13 * 7> a, at;
    std::array<float, 6 * 7 * 11> b, bt;
    std::array<float, 6 * 11> c;
    for (int h = 0; h != 6; ++h) {
        for (int i = 0; i != 13; ++i)
            for (int k = 0; k != 7; ++k)
                at[h * 91 + k * 13 + i] = a[h * 91 + i * 7 + k] = float((h * 11 + i * 3 + k * 7) % 17 - 8) / 8;
        for (int k = 0; k != 7; ++k)
            for (int j = 0; j != 11; ++j)
                bt[h * 77 + j * 7 + k] = b[h * 77 + k * 11 + j] = float((h * 7 + k * 5 + j * 3) % 19 - 9) / 8;
        for (int j = 0; j != 11; ++j)
            c[h * 11 + j] = float(h + j - 5) / 4;
    }
    std::array outputs{batched(a.data(), b.data(), c.data()), batch_nt(a.data(), bt.data(), c.data()),
                       batch_tn(at.data(), b.data(), c.data()), single(a.data(), b.data(), c.data()),
                       prefix(a.data(), bt.data(), c.data())};
    for (size_t t = 0; t != outputs.size(); ++t) {
        auto out   = outputs[t];
        auto count = t == 4 ? 6 : t == 3 ? 1 : 2;
        for (int h = 0; h != count; ++h)
            for (int i = 0; i != 13; ++i)
                for (int j = 0; j != 11; ++j) {
                    double expected = c[h * 11 + j];
                    for (int k = 0; k != 7; ++k)
                        expected += a[h * 91 + i * 7 + k] * b[h * 77 + k * 11 + j];
                    assert(std::abs(out[h * 143 + i * 11 + j] - expected) < 0.00001);
                }
        std::free(out);
    }
}
