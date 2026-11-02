#include <cassert>
#include <cmath>
#include <cstdlib>

#include <array>

extern "C" float* nn(float*, float*, float*);
extern "C" float* nt(float*, float*, float*);
extern "C" float* tn(float*, float*, float*);
extern "C" float* both(float*, float*, float*);

int main() {
    std::array<float, 13 * 7> a, at;
    std::array<float, 7 * 11> b, bt;
    std::array<float, 11> c;
    for (int i = 0; i != 13; ++i)
        for (int k = 0; k != 7; ++k)
            at[k * 13 + i] = a[i * 7 + k] = float((i * 3 + k * 7) % 17 - 8) / 8;
    for (int k = 0; k != 7; ++k)
        for (int j = 0; j != 11; ++j)
            bt[j * 7 + k] = b[k * 11 + j] = float((k * 5 + j * 3) % 19 - 9) / 8;
    for (int j = 0; j != 11; ++j)
        c[j] = float(j - 5) / 4;
    std::array outputs{nn(a.data(), b.data(), c.data()), nt(a.data(), bt.data(), c.data()),
                       tn(at.data(), b.data(), c.data()), both(at.data(), bt.data(), c.data())};
    for (auto out : outputs) {
        for (int i = 0; i != 13; ++i)
            for (int j = 0; j != 11; ++j) {
                double expected = c[j];
                for (int k = 0; k != 7; ++k)
                    expected += a[i * 7 + k] * b[k * 11 + j];
                expected = expected > 0 ? expected : 0;
                assert(std::abs(out[i * 11 + j] - expected) < 0.00001);
            }
        std::free(out);
    }
}
