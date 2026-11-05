#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct Pair {
    int64_t *full, *rows;
};
extern "C" int64_t* rows(int64_t, int64_t*);
extern "C" int64_t* grid(int64_t, int64_t*);
extern "C" Pair paired(int64_t, int64_t*);
extern "C" int64_t* nested(int64_t, int64_t*);
extern "C" int64_t* folded(int64_t);
extern "C" int64_t* fused(int64_t, int64_t*);
extern "C" int64_t* fused_grid(int64_t, int64_t*);

static int failures = 0;

static void expect(bool ok, const char* what, int64_t n, int64_t i) {
    if (ok) return;
    std::printf("%s: n=%lld i=%lld\n", what, (long long)n, (long long)i);
    ++failures;
}

static int64_t* input(int64_t cells) {
    auto x = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (cells + 1)));
    for (int64_t i = 0; i != cells; ++i)
        x[i] = 3 * i - 11;
    return x;
}

static void doubled(int64_t* (*f)(int64_t, int64_t*), const char* what, int64_t n, int64_t cols, int64_t plus = 0) {
    auto x = input(n * cols);
    auto y = f(n, x);
    for (int64_t i = 0; i != n * cols; ++i)
        expect(y[i] == 2 * x[i] + plus, what, n, i);
    std::free(y);
    std::free(x);
}

int main() {
    for (int64_t n = 0; n != 18; ++n) {
        doubled(rows, "rows", n, 3);
        doubled(grid, "grid", n, 7);
        doubled(nested, "nested", n, 6);
        doubled(fused, "fused", n, 3, 1);
        doubled(fused_grid, "fused_grid", n, 5, 1);

        auto x    = input(n * 3);
        auto pair = paired(n, x);
        for (int64_t i = 0; i != n; ++i) {
            int64_t sum = 0;
            for (int64_t j = 0; j != 3; ++j) {
                expect(pair.full[i * 3 + j] == 2 * x[i * 3 + j], "paired.full", n, i * 3 + j);
                sum += 2 * x[i * 3 + j];
            }
            expect(pair.rows[i] == sum, "paired.rows", n, i);
        }
        std::free(pair.full);
        std::free(pair.rows);
        std::free(x);

        int64_t seed = 3;
        for (int64_t k = 0; k != n; ++k)
            seed = seed * 10 + 2 * k;
        auto acc = folded(n);
        for (int64_t i = 0; i != 3; ++i)
            expect(acc[i] == seed, "folded", n, i);
        std::free(acc);
    }
    return failures != 0;
}
