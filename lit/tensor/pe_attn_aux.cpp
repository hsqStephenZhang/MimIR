#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" float* attn(float*, float*, float*);

int main() {
    const int L = 40, D = 16;
    std::vector<float> q(L * D), kt(D * L), v(L * D);
    for (int i = 0; i != L * D; ++i) {
        q[i]  = float(int(i * 7 % 13) - 6) / 8;
        kt[i] = float(int(i * 3 % 17) - 8) / 8;
        v[i]  = float(int(i * 5 % 11) - 5) / 4;
    }
    auto o  = attn(q.data(), kt.data(), v.data());
    int bad = 0;
    for (int i = 0; i != L; ++i) {
        std::vector<double> s(L);
        double m = -INFINITY, z = 0;
        for (int j = 0; j != L; ++j) {
            for (int d = 0; d != D; ++d)
                s[j] += double(q[i * D + d]) * kt[d * L + j];
            m = std::fmax(m, s[j]);
        }
        for (int j = 0; j != L; ++j)
            z += std::exp(s[j] - m);
        for (int d = 0; d != D; ++d) {
            double acc = 0;
            for (int j = 0; j != L; ++j)
                acc += std::exp(s[j] - m) / z * v[j * D + d];
            if (std::fabs(acc - o[i * D + d]) > 1e-3 * (1 + std::fabs(acc))) ++bad;
        }
    }
    if (bad) std::printf("attn: %d wrong\n", bad);
    std::free(o);
    return bad != 0;
}
