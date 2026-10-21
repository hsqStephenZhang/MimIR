// Driver for image_exec.mim: checks the inferred boxes against the values the maps actually take.
// Exit code: 0 if every box is sound and every exact coordinate is covered, 1 otherwise.

#include <cstdint>
#include <cstdio>

#include <array>
#include <set>

extern "C" {
// `mem.M` is erased and `Nat` is a 64-bit integer at the C ABI.
void img_win(int64_t* out, int64_t a0, int64_t a1);
void img_resh(int64_t* out, int64_t a0, int64_t a1);
void img_rdiv(int64_t* out, int64_t a0, int64_t a1);
void img_trs(int64_t* out, int64_t a0, int64_t a1);
}

using Map = std::array<int64_t, 2> (*)(int64_t, int64_t);

static std::array<int64_t, 2> win(int64_t oh, int64_t kh) { return {oh + kh, 2 * oh + kh}; }
static std::array<int64_t, 2> resh(int64_t i, int64_t j) {
    auto lin = i * 6 + j;
    return {(lin / 8) % 3, lin % 8};
}
static std::array<int64_t, 2> rdiv(int64_t i, int64_t j) { return {i % 8, j / 3}; }
static std::array<int64_t, 2> trs(int64_t i, int64_t j) { return {j + 2, i - 1}; }

static int check(const char* name,
                 void (*img)(int64_t*, int64_t, int64_t),
                 Map f,
                 std::array<int64_t, 2> ext,
                 std::array<int64_t, 2> shape,
                 std::array<int64_t, 2> a_lo,
                 std::array<int64_t, 2> a_hi) {
    int bad = 0;
    for (int64_t a0 = a_lo[0]; a0 < a_hi[0]; ++a0)
        for (int64_t a1 = a_lo[1]; a1 < a_hi[1]; ++a1) {
            int64_t r[6];
            img(r, a0, a1);
            std::set<int64_t> seen[2];
            for (int64_t i = a0; i < a0 + ext[0]; ++i)
                for (int64_t j = a1; j < a1 + ext[1]; ++j) {
                    auto v = f(i, j);
                    for (int c = 0; c < 2; ++c)
                        if (v[c] >= 0 && v[c] < shape[c]) seen[c].insert(v[c]);
                }
            for (int c = 0; c < 2; ++c) {
                auto lo = r[c], hi = r[c] + r[2 + c];
                for (auto v : seen[c])
                    if ((v < lo || v >= hi) && ++bad <= 5)
                        std::printf("%s @(%lld, %lld) coord %d: %lld outside [%lld, %lld)\n", name, (long long)a0,
                                    (long long)a1, c, (long long)v, (long long)lo, (long long)hi);
                if (r[4 + c])
                    for (auto v = lo; v < hi; ++v)
                        if (!seen[c].contains(v) && ++bad <= 5)
                            std::printf("%s @(%lld, %lld) coord %d: exact but %lld not reached\n", name, (long long)a0,
                                        (long long)a1, c, (long long)v);
            }
        }
    if (bad) std::printf("%s: %d failures\n", name, bad);
    return bad;
}

int main() {
    int bad = 0;
    bad += check("win", img_win, win, {3, 3}, {64, 64}, {0, 0}, {20, 20});
    bad += check("resh", img_resh, resh, {1, 4}, {3, 8}, {0, 0}, {4, 3});
    bad += check("rdiv", img_rdiv, rdiv, {5, 4}, {8, 64}, {0, 0}, {20, 40});
    bad += check("trs", img_trs, trs, {2, 3}, {16, 16}, {1, 0}, {16, 16});
    return bad ? 1 : 0;
}
