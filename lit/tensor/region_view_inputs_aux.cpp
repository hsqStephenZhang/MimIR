#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <array>

extern "C" float* tiled_views(float*, float*);
extern "C" float* multi_views(float*, float*);
extern "C" float* fold_views(float*, float*);
extern "C" float* points_views(float*, float*);

int main(int argc, char** argv) {
    float x[15], y[15];
    auto functions = std::array{tiled_views, multi_views, fold_views, points_views};
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (int i = 0; i != 15; ++i) {
            x[i] = i * 0.25f + repeat;
            y[i] = (17 - i) * 0.5f - repeat;
        }
        for (size_t variant = 0; variant != functions.size(); ++variant) {
            if (argc > 1 && variant != static_cast<size_t>(std::atoi(argv[1]))) continue;
            auto result = functions[variant](x, y);
            for (int i = 0; i != 5; ++i)
                for (int j = 0; j != 3; ++j) {
                    auto reference = (x[j * 5 + i] - y[j * 5 + i]) * (variant == 2 ? 2 : 1);
                    if (!std::isfinite(result[i * 3 + j]) || std::abs(result[i * 3 + j] - reference) > 1e-6f) {
                        std::fprintf(stderr, "region %zu cell %d/%d: %g != %g\n", variant, i, j, result[i * 3 + j],
                                     reference);
                        std::free(result);
                        return 1;
                    }
                }
            std::free(result);
        }
    }
}
