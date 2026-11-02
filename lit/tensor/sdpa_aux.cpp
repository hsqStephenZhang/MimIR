#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <limits>

extern "C" float* plain(float*, float*, float*);
extern "C" float* masked(float*, float*, float*, float*);

int main() {
    for (int scenario = 0; scenario != 4; ++scenario) {
        std::array<float, 30> q;
        std::array<float, 42> k;
        std::array<float, 56> v;
        std::array<float, 70> mask;
        for (size_t i = 0; i != q.size(); ++i)
            q[i] = float(int(i % 11) - 5) * (scenario == 1 ? 30.f : 0.3f);
        for (size_t i = 0; i != k.size(); ++i)
            k[i] = float(int(i % 13) - 6) * (scenario == 1 ? 20.f : 0.2f);
        for (size_t i = 0; i != v.size(); ++i)
            v[i] = float(int(i % 17) - 8) * 0.1f;
        for (int h = 0; h != 2; ++h)
            for (int i = 0; i != 5; ++i)
                for (int j = 0; j != 7; ++j)
                    mask[(h * 5 + i) * 7 + j] = (scenario == 2   ? j >= 3
                                                 : scenario == 3 ? i != 2
                                                                 : j <= i)
                                                  ? 0.f
                                                  : -std::numeric_limits<float>::infinity();
        for (bool causal : {false, true}) {
            auto out = causal ? masked(q.data(), k.data(), v.data(), mask.data()) : plain(q.data(), k.data(), v.data());
            for (int h = 0; h != 2; ++h)
                for (int i = 0; i != 5; ++i) {
                    std::array<float, 7> scores;
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j != 7; ++j) {
                        float score = 0;
                        for (int d = 0; d != 3; ++d)
                            score += q[(h * 5 + i) * 3 + d] * k[(h * 7 + j) * 3 + d];
                        scores[j] = score * 0.5f + (causal ? mask[(h * 5 + i) * 7 + j] : 0.f);
                        maximum   = std::max(maximum, scores[j]);
                    }
                    float sum = 0;
                    for (auto& score : scores) {
                        score = std::exp(score - maximum);
                        sum += score;
                    }
                    for (int d = 0; d != 4; ++d) {
                        float expected = 0;
                        for (int j = 0; j != 7; ++j)
                            expected += scores[j] / sum * v[(h * 7 + j) * 4 + d];
                        auto actual = out[(h * 5 + i) * 4 + d];
                        if (std::isnan(expected)
                                ? !std::isnan(actual)
                                : !std::isfinite(actual)
                                      || std::abs(actual - expected) > 2e-4f + 2e-4f * std::abs(expected)) {
                            std::fprintf(stderr, "causal=%d h=%d row=%d col=%d: %g != %g\n", causal, h, i, d, actual,
                                         expected);
                            return 1;
                        }
                    }
                }
            std::free(out);
        }
    }
}
