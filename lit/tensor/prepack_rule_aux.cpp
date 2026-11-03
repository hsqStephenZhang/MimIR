#include <cassert>
#include <cmath>
#include <cstdlib>

extern "C" float* gemm(float*, float*);
extern "C" float* batch(float*, float*);
extern "C" float* mutable_weight(float*, float*);
extern "C" float* prepared_a(float*, float*);
static int packs;
extern "C" void* record_pack(size_t alignment, size_t size) {
    ++packs;
    void* pointer;
    assert(posix_memalign(&pointer, alignment, size) == 0);
    return pointer;
}

void check(float* (*kernel)(float*, float*), int batches, bool frozen, float shift = 0) {
    float a[30], b[70], c[70];
    for (int i = 0; i != 30; ++i)
        a[i] = (i - 7) * 0.25f;
    for (int i = 0; i != 70; ++i) {
        b[i] = (i - 12) * 0.125f;
        c[i] = -b[i];
    }
    float* weights[] = {b, b, c, b};
    int expected[]   = {1, 1, 2, 3};
    int start        = packs;
    for (int call = 0; call != 4; ++call) {
        auto out = kernel(a, weights[call]);
        for (int z = 0; z != batches; ++z)
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 7; ++j) {
                    float reference = 0;
                    for (int k = 0; k != 5; ++k)
                        reference += (a[z * 15 + i * 5 + k] + shift) * weights[call][z * 35 + j * 5 + k];
                    assert(std::abs(out[z * 21 + i * 7 + j] - reference) < 1e-5f);
                }
        std::free(out);
#ifdef PREPACK_OFF
        assert(packs == 0);
#else
        assert(packs == start + (frozen ? expected[call] : 0));
#endif
    }
}

int main() {
    check(gemm, 1, true);
    check(batch, 2, true);
    check(mutable_weight, 1, false);
    check(prepared_a, 1, true, 0.125f);
}
