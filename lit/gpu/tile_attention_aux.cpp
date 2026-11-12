#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" float* attn(float*, float*, float*);

int main() {
    constexpr size_t H = 2, L = 64, D = 64;
    std::vector<float> q(H * L * D), k(H * L * D), v(H * L * D);
    for (size_t i = 0; i != q.size(); ++i) {
        q[i] = float(i % 97) / 97.0f - 0.5f;
        k[i] = float(i % 89) / 89.0f - 0.5f;
        v[i] = float(i % 83) / 83.0f - 0.5f;
    }
    auto o       = attn(q.data(), k.data(), v.data());
    int failures = 0;
    std::vector<double> s(L);
    for (size_t h = 0; h != H; ++h)
        for (size_t i = 0; i != L; ++i) {
            double mx = -1e30;
            for (size_t j = 0; j != L; ++j) {
                double acc = 0;
                for (size_t d = 0; d != D; ++d)
                    acc += double(q[(h * L + i) * D + d]) * k[(h * L + j) * D + d];
                mx = std::fmax(mx, s[j] = acc * 0.125);
            }
            double sum = 0;
            for (size_t j = 0; j != L; ++j)
                sum += (s[j] = std::exp(s[j] - mx));
            for (size_t d = 0; d != D; ++d) {
                double ref = 0;
                for (size_t j = 0; j != L; ++j)
                    ref += s[j] / sum * v[(h * L + j) * D + d];
                if (std::fabs(ref - o[(h * L + i) * D + d]) > 1e-3 * (1 + std::fabs(ref))) {
                    if (failures < 5) std::printf("o[%zu][%zu][%zu] = %g, expected %g\n", h, i, d, o[(h * L + i) * D + d], ref);
                    ++failures;
                }
            }
        }
    std::free(o);
    return failures ? 1 : 0;
}
