#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>

extern "C" {
float* sources_denote(float*, float*);
float* sources_staged(float*, float*);
float* sources_unstaged(float*, float*);
}

int main() {
    constexpr int m = 32, k = 16, n = 32;
    float x[m * k], y[k * n];
    for (int i = 0; i != m * k; ++i)
        x[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != k * n; ++i)
        y[i] = float(i % 5) * 0.5f - 1.0f;
    int failures = 0;
    for (auto gemm : {sources_denote, sources_staged, sources_unstaged}) {
        auto c = gemm(x, y);
        for (int i = 0; i != m; ++i)
            for (int j = 0; j != n; ++j) {
                float ref = 0.0f;
                for (int l = 0; l != k; ++l)
                    ref += 2.0f * x[i * k + l] * (y[l * n + j] + 1.0f);
                if (std::fabs(ref - c[i * n + j]) > 1e-3f * (1.0f + std::fabs(ref)) && failures++ < 5)
                    std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
            }
        std::free(c);
    }
    return failures ? 1 : 0;
}
