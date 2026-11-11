#include <cassert>
#include <cmath>
#include <cstdlib>

#include <array>
#include <vector>
extern "C" float* layout_00(float*, float*);
extern "C" float* layout_01(float*, float*);
extern "C" float* layout_10(float*, float*);
extern "C" float* layout_11(float*, float*);
int main() {
    std::array kernels{layout_00, layout_01, layout_10, layout_11};
    for (int ac = 0; ac < 2; ++ac)
        for (int bc = 0; bc < 2; ++bc) {
            int ar = ac ? 37 : 13, ax = ac ? 13 : 37;
            int br = bc ? 19 : 37, bx = bc ? 37 : 19;
            std::vector<float> a((ar + 2) * (ax + 3), 1000), b((br + 2) * (bx + 5), 1000);
            auto av = [](int i, int k) { return float((i * 3 + k * 7) % 17 - 8) / 8; };
            auto bv = [](int k, int j) { return float((k * 5 + j * 3) % 19 - 9) / 8; };
            for (int i = 0; i < 13; ++i)
                for (int k = 0; k < 37; ++k)
                    a[((ac ? k : i) + 1) * (ax + 3) + (ac ? i : k) + 1] = av(i, k);
            for (int k = 0; k < 37; ++k)
                for (int j = 0; j < 19; ++j)
                    b[((bc ? j : k) + 1) * (bx + 5) + (bc ? k : j) + 2] = bv(k, j);
            auto out = kernels[ac * 2 + bc](a.data(), b.data());
            for (int i = 0; i < 13; ++i)
                for (int j = 0; j < 19; ++j) {
                    double expected = 0;
                    for (int k = 0; k < 37; ++k)
                        expected += av(i, k) * bv(k, j);
                    assert(std::abs(out[j * 13 + i] - expected) < 0.00001);
                }
            std::free(out);
        }
}
