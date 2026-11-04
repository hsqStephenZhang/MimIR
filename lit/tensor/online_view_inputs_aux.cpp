#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <array>

extern "C" float* qk(float*, float*, float*);
extern "C" float* kv(float*, float*, float*);
extern "C" float* qv(float*, float*, float*);
extern "C" float* qkv(float*, float*, float*);

int main(int argc, char** argv) {
    std::array<float, 136> q, k, v;
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (int i = 0; i != 136; ++i) {
            q[i] = ((i * 13 + repeat * 7) % 37 - 18) * 0.07f;
            k[i] = ((i * 7 + repeat * 11) % 43 - 21) * 0.09f;
            v[i] = ((i * 11 + repeat * 3) % 47 - 23) * 0.11f;
        }
        int flags[]    = {3, 6, 5, 7};
        auto functions = std::array{qk, kv, qv, qkv};
        for (size_t variant = 0; variant != functions.size(); ++variant) {
            if (argc > 1 && variant != static_cast<size_t>(std::atoi(argv[1]))) continue;
            auto result = functions[variant](q.data(), k.data(), v.data());
            auto read   = [&](auto& input, int bit, int h, int row, int d) {
                return input[(flags[variant] & bit) ? (row * 2 + h) * 4 + d : (h * 17 + row) * 4 + d];
            };
            for (int h = 0; h != 2; ++h)
                for (int i = 0; i != 17; ++i) {
                    std::array<double, 17> scores;
                    double largest = -INFINITY, sum = 0;
                    for (int j = 0; j != 17; ++j) {
                        double dot = 0;
                        for (int d = 0; d != 4; ++d)
                            dot += read(q, 1, h, i, d) * read(k, 2, h, j, d);
                        scores[j] = dot * 0.5;
                        largest   = std::max(largest, scores[j]);
                    }
                    for (auto& score : scores) {
                        score = std::exp(score - largest);
                        sum += score;
                    }
                    for (int d = 0; d != 4; ++d) {
                        double reference = 0;
                        for (int j = 0; j != 17; ++j)
                            reference += scores[j] / sum * read(v, 4, h, j, d);
                        auto actual = result[(h * 17 + i) * 4 + d];
                        if (!std::isfinite(actual) || std::abs(actual - reference) > 2e-5) {
                            std::fprintf(stderr, "view mask %d cell %d/%d/%d: %.9g != %.9g\n", flags[variant], h, i, d,
                                         actual, reference);
                            std::free(result);
                            return 1;
                        }
                    }
                }
            std::free(result);
        }
    }
}
