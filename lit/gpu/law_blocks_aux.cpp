#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>
#include <vector>

constexpr int H = 2, L = 128, D = 64;

#ifndef WARPS_ONLY
extern "C" float* attn_whole(float*, float*, float*);
extern "C" float* attn_serial(float*, float*, float*);
#endif
extern "C" float* attn_warps(float*, float*, float*);

int main() {
    std::vector<float> q(H * L * D), k(H * L * D), v(H * L * D), want(H * L * D);
    for (int i = 0; i != H * L * D; ++i) {
        q[i] = float(i % 97) / 97.0f - 0.5f;
        k[i] = float(i % 89) / 89.0f - 0.5f;
        v[i] = float(i % 83) / 83.0f - 0.5f;
    }
    for (int h = 0; h != H; ++h)
        for (int i = 0; i != L; ++i) {
            double s[L], mx = -1e30, sum = 0;
            for (int j = 0; j != L; ++j) {
                s[j] = 0;
                for (int d = 0; d != D; ++d)
                    s[j] += double(q[(h * L + i) * D + d]) * k[(h * L + j) * D + d];
                s[j] *= 0.125;
                mx = std::fmax(mx, s[j]);
            }
            for (int j = 0; j != L; ++j)
                sum += (s[j] = std::exp(s[j] - mx));
            for (int d = 0; d != D; ++d) {
                double o = 0;
                for (int j = 0; j != L; ++j)
                    o += s[j] / sum * v[(h * L + j) * D + d];
                want[(h * L + i) * D + d] = float(o);
            }
        }
    struct Case {
        float* (*f)(float*, float*, float*);
        const char* what;
    };
    int failures = 0;
    auto cases   = {
#ifndef WARPS_ONLY
        Case{ attn_whole, "meaning"},
        Case{attn_serial,  "serial"},
#endif
        Case{ attn_warps,   "warps"},
    };
    for (auto [f, what] : cases) {
        auto got = f(q.data(), k.data(), v.data());
        for (int i = 0; i != H * L * D; ++i)
            if (std::fabs(got[i] - want[i]) > 1e-3f * (1 + std::fabs(want[i]))) {
                std::printf("%s: element %d is %f, not %f\n", what, i, got[i], want[i]);
                ++failures;
                break;
            }
    }
    return failures != 0;
}
