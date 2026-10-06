// Driver for dot_par_exec.mim: calls the LLVM-compiled products and checks every output against a scalar reference.
// Exit code: 0 if all outputs match within tolerance, 1 otherwise.

#include <cmath>
#include <cstdio>

#include <vector>

extern "C" {
float* mm(float* a, float* b);
float* mm_tail(float* a, float* b);
}

static int check(const char* name, float* (*f)(float*, float*), size_t M, size_t K, size_t N) {
    std::vector<float> a(M * K), b(K * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(i % 13) / 13.0f - 0.4f;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>((7 * i) % 11) / 11.0f - 0.5f;

    float* out = f(a.data(), b.data());

    int bad = 0;
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k)
                s += static_cast<double>(a[m * K + k]) * b[k * N + n];
            if (std::fabs(out[m * N + n] - s) > 1e-3) {
                if (bad < 5) std::printf("%s[%zu][%zu] = %f, want %f\n", name, m, n, out[m * N + n], s);
                ++bad;
            }
        }
    return bad;
}

int main() {
    int bad = 0;
    bad += check("mm", mm, 64, 256, 512);
    bad += check("mm_tail", mm_tail, 50, 128, 200);
    return bad ? 1 : 0;
}
