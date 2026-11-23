#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

extern "C" {
float* row_sum_f32(float*);
float* row_max_f32(float*);
double* row_sum_f64(double*);
float* gemv_f32(float*, float*);
int16_t* row_sum_i16(int16_t*);
}

int main() {
    int bad = 0;

    // 1. row_sum_f32 (17 x 513)
    {
        int M = 17, K = 513;
        std::vector<float> x(M * K);
        for (int i = 0; i < M * K; ++i) x[i] = float((i * 7) % 19 - 9) * 0.25f;
        float* got = row_sum_f32(x.data());
        for (int m = 0; m < M; ++m) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += x[m * K + k];
            if (std::fabs(got[m] - ref) > 1e-3) {
                std::fprintf(stderr, "row_sum_f32[%d]: got %g want %g\n", m, got[m], ref);
                ++bad;
            }
        }
        std::free(got);
    }

    // 2. row_max_f32 (13 x 37)
    {
        int M = 13, K = 37;
        std::vector<float> x(M * K);
        for (int i = 0; i < M * K; ++i) x[i] = float((i * 11) % 29 - 14) * 0.5f;
        float* got = row_max_f32(x.data());
        for (int m = 0; m < M; ++m) {
            float ref = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < K; ++k) ref = std::fmax(ref, x[m * K + k]);
            if (got[m] != ref) {
                std::fprintf(stderr, "row_max_f32[%d]: got %g want %g\n", m, got[m], ref);
                ++bad;
            }
        }
        std::free(got);
    }

    // 3. row_sum_f64 (19 x 130)
    {
        int M = 19, K = 130;
        std::vector<double> x(M * K);
        for (int i = 0; i < M * K; ++i) x[i] = double((i * 5) % 23 - 11) * 0.125;
        double* got = row_sum_f64(x.data());
        for (int m = 0; m < M; ++m) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += x[m * K + k];
            if (std::fabs(got[m] - ref) > 1e-9) {
                std::fprintf(stderr, "row_sum_f64[%d]: got %g want %g\n", m, got[m], ref);
                ++bad;
            }
        }
        std::free(got);
    }

    // 4. gemv_f32 (64 x 300) * (300 x 1)
    {
        int M = 64, K = 300;
        std::vector<float> a(M * K), v(K);
        for (int i = 0; i < M * K; ++i) a[i] = float((i * 3) % 11 - 5) * 0.25f;
        for (int k = 0; k < K; ++k) v[k] = float((k * 7) % 13 - 6) * 0.25f;
        float* got = gemv_f32(a.data(), v.data());
        for (int m = 0; m < M; ++m) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += double(a[m * K + k]) * double(v[k]);
            if (std::fabs(got[m] - ref) > 1e-3) {
                std::fprintf(stderr, "gemv_f32[%d]: got %g want %g\n", m, got[m], ref);
                ++bad;
            }
        }
        std::free(got);
    }

    // 5. row_sum_i16 (23 x 275)
    {
        int M = 23, K = 275;
        std::vector<int16_t> x(M * K);
        for (int i = 0; i < M * K; ++i) x[i] = int16_t((i * 5) % 17 - 8);
        int16_t* got = row_sum_i16(x.data());
        for (int m = 0; m < M; ++m) {
            int16_t ref = 0;
            for (int k = 0; k < K; ++k) ref = int16_t(ref + x[m * K + k]);
            if (got[m] != ref) {
                std::fprintf(stderr, "row_sum_i16[%d]: got %d want %d\n", m, int(got[m]), int(ref));
                ++bad;
            }
        }
        std::free(got);
    }

    return bad != 0;
}
