#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct Pair {
    int64_t *root, *member;
};
extern "C" Pair sh2(int64_t*, int64_t*);
extern "C" Pair sh2_sym(int64_t, int64_t*, int64_t*);
extern "C" Pair sh2_cap(int64_t, int64_t*, int64_t*);
extern "C" Pair gemm_yield(int64_t, int64_t*, int64_t*, int64_t*);
extern "C" Pair unaligned(int64_t*, int64_t*);

static int failures  = 0;
static int64_t calls = 0;

extern "C" int64_t twice_count(int64_t x) {
    ++calls;
    return 2 * x;
}

static int64_t* fill(int64_t cells, int64_t scale, int64_t mod, int64_t shift) {
    auto x = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (cells + 1)));
    for (int64_t i = 0; i != cells; ++i)
        x[i] = scale * i % mod + shift;
    return x;
}

static void fail(const char* what, int64_t n, int64_t i) {
    std::printf("%s: n=%lld at %lld\n", what, (long long)n, (long long)i);
    ++failures;
}

/// Checks `(2x + 1 + y, 2x)` and, when counting, that `2x` is computed once per element.
static void
shared(const char* what, int64_t n, Pair (*run)(int64_t*, int64_t*), Pair (*sym)(int64_t, int64_t*, int64_t*)) {
    auto x = fill(n * 4, 3, 1000, -7), y = fill(n * 4, 5, 11, 0);
    calls  = 0;
    auto r = run ? run(x, y) : sym(n, x, y);
#ifdef COUNT
    if (calls != n * 4) fail(what, n, calls);
#endif
    for (int64_t i = 0; i != n * 4; ++i)
        if (r.member[i] != 2 * x[i] || r.root[i] != 2 * x[i] + 1 + y[i]) fail(what, n, i);
    std::free(x);
    std::free(y);
    std::free(r.root);
    std::free(r.member);
}

/// Checks `(a·b·2 + relu(c), relu(c))`.
static void gemm(int64_t m) {
    auto a = fill(m * 7, 5, 17, -8), b = fill(42, 3, 17, -4), c = fill(m * 6, 7, 13, -6);
    auto r = gemm_yield(m, a, b, c);
    for (int64_t i = 0; i != m * 6; ++i) {
        int64_t acc = 0;
        for (int64_t l = 0; l != 7; ++l)
            acc += a[i / 6 * 7 + l] * b[l * 6 + i % 6];
        auto q = c[i] < 0 ? 0 : c[i];
        if (r.member[i] != q || r.root[i] != acc * 2 + q) fail("gemm_yield", m, i);
    }
    std::free(a);
    std::free(b);
    std::free(c);
    std::free(r.root);
    std::free(r.member);
}

/// Checks `(2xᵀ + y, 2x)`.
static void transposed() {
    auto x = fill(16, 3, 1000, -7), y = fill(16, 5, 11, 0);
    auto r = unaligned(x, y);
    for (int64_t i = 0; i != 16; ++i)
        if (r.member[i] != 2 * x[i] || r.root[i] != 2 * x[i % 4 * 4 + i / 4] + y[i]) fail("unaligned", 4, i);
    std::free(x);
    std::free(y);
    std::free(r.root);
    std::free(r.member);
}

int main() {
    shared("sh2", 6, sh2, nullptr);
    for (int64_t n = 0; n != 12; ++n) {
        shared("sh2_sym", n, nullptr, sh2_sym);
        shared("sh2_cap", n, nullptr, sh2_cap);
#ifndef COUNT
        gemm(n);
#endif
    }
#ifndef COUNT
    transposed();
#endif
    return failures != 0;
}
