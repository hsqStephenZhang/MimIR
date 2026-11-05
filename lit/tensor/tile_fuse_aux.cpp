#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* chain(int64_t*, int64_t*);
extern "C" int64_t* chain_sym(int64_t, int64_t*, int64_t*);
extern "C" int64_t* excluded(int64_t*, int64_t*);
extern "C" int64_t* rejected(int64_t*, int64_t*);
extern "C" int64_t* rows(int64_t*, int64_t*);

struct Pair {
    int64_t *root, *p1;
};
extern "C" Pair shared(int64_t*, int64_t*);
extern "C" Pair shared_sym(int64_t, int64_t*, int64_t*);

static int failures = 0;

static void expect(bool ok, const char* what, int64_t n, int64_t i) {
    if (ok) return;
    std::printf("%s: n=%lld i=%lld\n", what, (long long)n, (long long)i);
    ++failures;
}

static int64_t* fill(int64_t cells, int64_t scale, int64_t shift) {
    auto x = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (cells + 1)));
    for (int64_t i = 0; i != cells; ++i)
        x[i] = scale * i + shift;
    return x;
}

static void
chained(int64_t* (*f)(int64_t*, int64_t*), int64_t* (*g)(int64_t, int64_t*, int64_t*), const char* what, int64_t n) {
    auto x = fill(n * 4, 3, -11), y = fill(n * 4, -2, 5);
    auto z = f ? f(x, y) : g(n, x, y);
    for (int64_t i = 0; i != n * 4; ++i)
        expect(z[i] == 2 * x[i] + 1 + y[i], what, n, i);
    std::free(x);
    std::free(y);
    std::free(z);
}

int main() {
    chained(chain, nullptr, "chain", 6);
    chained(excluded, nullptr, "excluded", 6);
    chained(rejected, nullptr, "rejected", 6);
    for (int64_t n = 0; n != 10; ++n)
        chained(nullptr, chain_sym, "chain_sym", n);

    for (int64_t n = 0; n != 10; ++n) {
        auto x = fill(n * 4, 3, -11), y = fill(n * 4, -2, 5);
        auto pair = n == 6 ? shared(x, y) : shared_sym(n, x, y);
        for (int64_t i = 0; i != n * 4; ++i) {
            expect(pair.root[i] == 2 * x[i] + 1 + y[i], "shared.root", n, i);
            expect(pair.p1[i] == 2 * x[i], "shared.p1", n, i);
        }
        std::free(pair.root);
        std::free(pair.p1);
        std::free(x);
        std::free(y);
    }

    auto a = fill(15, 5, -30), b = fill(12, -1, 4);
    auto c = rows(a, b);
    for (int64_t i = 0; i != 5; ++i)
        for (int64_t j = 0; j != 4; ++j) {
            int64_t acc = 0;
            for (int64_t k = 0; k != 3; ++k)
                acc += (a[i * 3 + k] < 0 ? 0 : a[i * 3 + k]) * b[k * 4 + j];
            expect(c[i * 4 + j] == acc, "rows", 5, i * 4 + j);
        }
    std::free(a);
    std::free(b);
    std::free(c);
    return failures != 0;
}
