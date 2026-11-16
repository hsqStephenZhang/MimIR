#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>

extern "C" {
float* post_denote(float*, float*, float*);
float* post_staged(float*, float*, float*);
float* post_unstaged(float*, float*, float*);
float* post_req(float*, float*, float*);
int64_t* rows_run(int64_t*);
int64_t* rows_req(int64_t*);
}

int main() {
    constexpr int m = 32, k = 16, n = 32;
    float x[m * k], y[k * n], bias[n];
    for (int i = 0; i != m * k; ++i)
        x[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != k * n; ++i)
        y[i] = float(i % 5) * 0.5f - 1.0f;
    for (int j = 0; j != n; ++j)
        bias[j] = float(j) * 0.25f;
    int failures = 0;
    for (auto gemm : {post_denote, post_staged, post_unstaged, post_req}) {
        auto c = gemm(x, y, bias);
        for (int i = 0; i != m; ++i)
            for (int j = 0; j != n; ++j) {
                float ref = bias[j];
                for (int l = 0; l != k; ++l)
                    ref += x[i * k + l] * y[l * n + j];
                if (std::fabs(ref - c[i * n + j]) > 1e-3f * (1.0f + std::fabs(ref)) && failures++ < 5)
                    std::printf("c[%d][%d] = %g, expected %g\n", i, j, c[i * n + j], ref);
            }
        std::free(c);
    }
    int64_t z[16 * 16];
    for (int i = 0; i != 16 * 16; ++i)
        z[i] = i % 11 - 5;
    for (auto rows : {rows_run, rows_req}) {
        auto r = rows(z);
        for (int i = 0; i != 16; ++i) {
            int64_t ref = 16;
            for (int j = 0; j != 16; ++j)
                ref += z[i * 16 + j];
            if (r[i] != ref && failures++ < 5)
                std::printf("r[%d] = %lld, expected %lld\n", i, (long long)r[i], (long long)ref);
        }
        std::free(r);
    }
    return failures ? 1 : 0;
}
