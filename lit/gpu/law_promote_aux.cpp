#include <cmath>
#include <cstdio>

#include <initializer_list>

extern "C" float* pro_serial_run(float*, float*);
extern "C" float* pro_serial_req(float*, float*);
extern "C" float* pro_group_run(float*, float*);
extern "C" float* pro_group_req(float*, float*);
extern "C" float* pro_items_run(float*, float*);
extern "C" float* pro_items_req(float*, float*);

int main() {
    constexpr int m = 16, k = 20, n = 16;
    float a[m * k], b[k * n];
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i % 7) - 3.0f;
    for (int i = 0; i != k * n; ++i)
        b[i] = float(i % 5) * 0.5f - 1.0f;
    for (auto f : {pro_serial_run, pro_serial_req, pro_group_run, pro_group_req, pro_items_run, pro_items_req}) {
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
