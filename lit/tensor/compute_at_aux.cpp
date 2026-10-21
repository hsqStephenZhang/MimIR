// Driver for compute_at_exec.mim: calls the kernels whose producers are computed one tile at a time and checks
// every output against a scalar reference.
// Exit code: 0 if all outputs match, 1 otherwise.

#include <cstdint>
#include <cstdio>

extern "C" {
// `«s; I32»` tensors lower to plain pointers at the C ABI.
int32_t* mm_at0(int32_t* a, int32_t* b);
int32_t* mm_at1(int32_t* a, int32_t* b);
int32_t* conv_at(int32_t* x, int32_t* w);
int32_t* red_at(int32_t* a, int32_t* b);
}

static int compare(const char* name, const int32_t* got, const int32_t* want, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (got[i] != want[i] && ++bad <= 5) std::printf("%s[%d]: got %d want %d\n", name, i, got[i], want[i]);
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    int32_t a[4 * 6], b[6 * 5], b2[4 * 5], x[2 * 2 * 6 * 6], w[3 * 2 * 3 * 3];
    for (int i = 0; i < 4 * 6; ++i)
        a[i] = (i * 7) % 11 - 5;
    for (int i = 0; i < 6 * 5; ++i)
        b[i] = (i * 5) % 13 - 6;
    for (int i = 0; i < 4 * 5; ++i)
        b2[i] = (i * 3) % 7 - 3;
    for (int i = 0; i < 2 * 2 * 6 * 6; ++i)
        x[i] = (i * 11) % 9 - 4;
    for (int i = 0; i < 3 * 2 * 3 * 3; ++i)
        w[i] = (i * 13) % 5 - 2;

    int32_t mm[4 * 5];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j) {
            int32_t s = 0;
            for (int k = 0; k < 6; ++k)
                s += (a[i * 6 + k] + 1) * b[k * 5 + j];
            mm[i * 5 + j] = s;
        }

    int32_t cv[2 * 3 * 4 * 4];
    for (int n = 0; n < 2; ++n)
        for (int c = 0; c < 3; ++c)
            for (int oh = 0; oh < 4; ++oh)
                for (int ow = 0; ow < 4; ++ow) {
                    int32_t s = 0;
                    for (int ci = 0; ci < 2; ++ci)
                        for (int kh = 0; kh < 3; ++kh)
                            for (int kw = 0; kw < 3; ++kw)
                                s += (x[((n * 2 + ci) * 6 + oh + kh) * 6 + ow + kw] + 1)
                                   * w[((c * 2 + ci) * 3 + kh) * 3 + kw];
                    cv[((n * 3 + c) * 4 + oh) * 4 + ow] = s;
                }

    int32_t rd[4 * 5];
    for (int i = 0; i < 4; ++i) {
        int32_t s = 0;
        for (int k = 0; k < 6; ++k)
            s += a[i * 6 + k];
        for (int j = 0; j < 5; ++j)
            rd[i * 5 + j] = s * b2[i * 5 + j];
    }

    int bad = 0;
    bad += compare("mm_at0", mm_at0(a, b), mm, 4 * 5);
    bad += compare("mm_at1", mm_at1(a, b), mm, 4 * 5);
    bad += compare("conv_at", conv_at(x, w), cv, 2 * 3 * 4 * 4);
    bad += compare("red_at", red_at(a, b2), rd, 4 * 5);
    return bad ? 1 : 0;
}
