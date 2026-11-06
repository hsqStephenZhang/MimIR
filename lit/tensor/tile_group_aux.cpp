#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* gemm(int64_t, int64_t*, int64_t*);
extern "C" int64_t* rows(int64_t, int64_t*);
extern "C" int64_t* chain(int64_t, int64_t*);

static int failures = 0;

static int64_t* fill(int64_t cells, int64_t scale, int64_t shift) {
    auto x = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (cells + 1)));
    for (int64_t i = 0; i != cells; ++i)
        x[i] = scale * i % 17 + shift;
    return x;
}

static void expect(bool ok, const char* what, int64_t m, int64_t i) {
    if (ok) return;
    std::printf("%s: m=%lld at %lld\n", what, (long long)m, (long long)i);
    ++failures;
}

int main() {
    for (int64_t m = 0; m != 11; ++m) {
        auto a = fill(m * 70, 5, -8), b = fill(70 * 20, 3, -4), x = fill(m * 9, 7, -5);
        auto c = gemm(m, a, b);
        for (int64_t i = 0; i != m * 20; ++i) {
            int64_t acc = 0;
            for (int64_t k = 0; k != 70; ++k) {
                auto v = a[i / 20 * 70 + k];
                acc += (v < 0 ? 0 : v) * b[k * 20 + i % 20];
            }
            expect(c[i] == acc, "gemm", m, i);
        }
        auto s = rows(m, x);
        for (int64_t i = 0; i != m; ++i) {
            int64_t acc = 0;
            for (int64_t j = 0; j != 9; ++j)
                acc += 2 * x[i * 9 + j];
            expect(s[i] == acc, "rows", m, i);
        }
        auto t = chain(m, x);
        for (int64_t i = 0; i != m * 9; ++i)
            expect(t[i] == 2 * x[i] + 1, "chain", m, i);
        std::free(a);
        std::free(b);
        std::free(x);
        std::free(c);
        std::free(s);
        std::free(t);
    }
    return failures != 0;
}
