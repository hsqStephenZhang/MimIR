// Driver for tile_at_exec.mim: computes boxes of C = A · B at every aligned run-time origin and checks
// each against a scalar reference.
// Exit code: 0 if all boxes match, 1 otherwise.

#include <cstdint>
#include <cstdio>

extern "C" {
// `«s; I32»` tensors lower to plain pointers and `Nat` to a 64-bit integer at the C ABI.
int32_t* mm_tile(int32_t* a, int32_t* b, int64_t i0, int64_t j0);
int32_t* mmt_tile(int32_t* a, int32_t* b, int64_t j0, int64_t i0);
int32_t* mm_tile2(int32_t* a, int32_t* b, int64_t i0, int64_t j0);
int32_t* mm_sched(int32_t* a, int32_t* b, int64_t i0, int64_t j0);
}

constexpr int M = 6, K = 4, N = 8;

static int32_t a[M * K], b[K * N];

static int32_t ref(int i, int j) {
    int32_t s = 0;
    for (int k = 0; k < K; ++k)
        s += a[i * K + k] * b[k * N + j];
    return s;
}

// The box of C at (i0, j0) with extents (TI, TJ), transposed when `t` is set.
static int check(const char* name, int32_t* (*f)(int32_t*, int32_t*, int64_t, int64_t), int TI, int TJ, bool t) {
    int bad = 0;
    for (int i0 = 0; i0 + TI <= M; i0 += TI)
        for (int j0 = 0; j0 + TJ <= N; j0 += TJ) {
            int32_t* out = t ? f(a, b, j0, i0) : f(a, b, i0, j0);
            for (int i = 0; i < TI; ++i)
                for (int j = 0; j < TJ; ++j) {
                    int32_t got = t ? out[j * TI + i] : out[i * TJ + j];
                    int32_t want = ref(i0 + i, j0 + j);
                    if (got != want && ++bad <= 5)
                        std::printf("%s @(%d, %d) [%d, %d]: got %d want %d\n", name, i0, j0, i, j, got, want);
                }
        }
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    for (int i = 0; i < M * K; ++i)
        a[i] = (i * 7) % 11 - 5;
    for (int i = 0; i < K * N; ++i)
        b[i] = (i * 5) % 13 - 6;

    int bad = 0;
    bad += check("mm_tile", mm_tile, 2, 4, false);
    bad += check("mmt_tile", mmt_tile, 3, 4, true);
    bad += check("mm_tile2", mm_tile2, 2, 4, false);
    bad += check("mm_sched", mm_sched, 2, 4, false);
    return bad ? 1 : 0;
}
