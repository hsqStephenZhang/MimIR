#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>

constexpr int H = 2, M = 5, S = 7, K = 3, N = 4;
struct Acc {
    float m, l, o[N];
};

extern "C" Acc* fa_denote(float*, float*, float*);
extern "C" Acc* fa_blocks(float*, float*, float*);
extern "C" Acc* fa_split(float*, float*, float*);
extern "C" Acc* fa_rows(float*, float*, float*);
extern "C" Acc* fa_items(float*, float*, float*);

int main() {
    float q[H * M * K], k[H * S * K], v[H * S * N], want[H * M * N];
    for (int i = 0; i != H * M * K; ++i)
        q[i] = float((i * 5) % 7) * 0.5f - 1.5f;
    for (int i = 0; i != H * S * K; ++i)
        k[i] = float((i * 3) % 11) * 0.25f - 1.0f;
    for (int i = 0; i != H * S * N; ++i)
        v[i] = float((i * 7) % 13) * 0.3f - 2.0f;
    for (int h = 0; h != H; ++h)
        for (int i = 0; i != M; ++i) {
            double s[S], mx = -1e30, sum = 0;
            for (int j = 0; j != S; ++j) {
                s[j] = 0;
                for (int d = 0; d != K; ++d)
                    s[j] += q[(h * M + i) * K + d] * k[(h * S + j) * K + d];
                s[j] *= 0.5;
                mx = std::fmax(mx, s[j]);
            }
            for (int j = 0; j != S; ++j)
                sum += std::exp(s[j] - mx);
            for (int n = 0; n != N; ++n) {
                double o = 0;
                for (int j = 0; j != S; ++j)
                    o += std::exp(s[j] - mx) / sum * v[(h * S + j) * N + n];
                want[(h * M + i) * N + n] = float(o);
            }
        }
    struct Case {
        Acc* (*f)(float*, float*, float*);
        const char* what;
    };
    int failures = 0;
    auto cases   = {
        Case{fa_denote,         "denote"},
        Case{fa_blocks,      "kv blocks"},
        Case{ fa_split,       "split kv"},
        Case{  fa_rows,           "rows"},
        Case{ fa_items, "split kv items"},
    };
    for (auto [f, what] : cases) {
        auto got = f(q, k, v);
        for (int r = 0; r != H * M; ++r)
            for (int n = 0; n != N; ++n)
                if (std::fabs(got[r].o[n] / got[r].l - want[r * N + n]) > 1e-4f) {
                    std::printf("%s: row %d col %d is %f, not %f\n", what, r, n, got[r].o[n] / got[r].l,
                                want[r * N + n]);
                    ++failures;
                    goto next;
                }
next:
        std::free(got);
    }
    return failures != 0;
}
