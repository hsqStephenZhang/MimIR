#include <cmath>
#include <cstdlib>
#include <vector>

extern "C" float* view_a(float*, float*);

int main() {
    constexpr int M = 13, K = 300, N = 37;
    std::vector<float> a(K * M), b(K * N);
    for (size_t i = 0; i != a.size(); ++i) a[i] = float(int(i % 13) - 6) / 8;
    for (size_t i = 0; i != b.size(); ++i) b[i] = float(int(i % 7) - 3) / 4;
    auto c = view_a(a.data(), b.data());
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j) {
            double ref = 0;
            for (int k = 0; k != K; ++k) ref += double(a[k * M + i]) * b[k * N + j];
            if (std::abs(c[i * N + j] - ref) > 1e-3) return 1;
        }
    std::free(c);
}
