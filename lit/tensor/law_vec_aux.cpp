#include <cmath>
#include <cstdio>

#include <initializer_list>

extern "C" float* vec_run(float*, float*);
extern "C" float* vec_req(float*, float*);
extern "C" float* cols_run(float*, float*);
extern "C" float* cols_req(float*, float*);
int main() {
    constexpr int m = 16, k = 20, n = 16;
    float a[m * k], b[k * n];
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != k * n; ++i)
        b[i] = float(i % 5) * 0.5f - 1.0f;
    for (auto f : {vec_run, vec_req, cols_run, cols_req}) {
        auto c = f(a, b);
        for (int i = 0; i != m; ++i)
            for (int j = 0; j != n; ++j) {
                float want = 0;
                for (int l = 0; l != k; ++l)
                    want += a[i * k + l] * b[l * n + j];
                if (std::fabs(c[i * n + j] - want) > 1e-3f) {
                    std::printf("c[%d][%d] = %f, not %f\n", i, j, c[i * n + j], want);
                    return 1;
                }
            }
    }
    return 0;
}
