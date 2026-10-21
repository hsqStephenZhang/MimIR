// Driver for image_exec.mim: checks the inferred boxes against the values the maps actually take.
// Exit code: 0 if every box is sound, fits its cap, and is covered where exact; 1 otherwise.

#include <cstdint>
#include <cstdio>

#include <array>
#include <set>
#include <vector>

extern "C" {
// `mem.M` is erased and `Nat` is a 64-bit integer at the C ABI.
void img_win(int64_t* out, int64_t a0, int64_t a1);
void img_resh(int64_t* out, int64_t a0, int64_t a1);
void img_rdiv(int64_t* out, int64_t a0, int64_t a1);
void img_trs(int64_t* out, int64_t a0, int64_t a1);
void img_st(int64_t* out, int64_t a0, int64_t a1);
void pre_ws(int64_t* out, int64_t y0, int64_t y1);
}

using Map  = std::array<int64_t, 2> (*)(int64_t, int64_t);
using Maps = std::vector<Map>;

static std::array<int64_t, 2> win(int64_t oh, int64_t kh) { return {oh + kh, 2 * oh + kh}; }
static std::array<int64_t, 2> resh(int64_t i, int64_t j) {
    auto lin = i * 6 + j;
    return {(lin / 8) % 3, lin % 8};
}
static std::array<int64_t, 2> rdiv(int64_t i, int64_t j) { return {i % 8, j / 3}; }
static std::array<int64_t, 2> trs(int64_t i, int64_t j) { return {j + 2, i - 1}; }
static std::array<int64_t, 2> st0(int64_t i, int64_t j) { return {i - 1, j}; }
static std::array<int64_t, 2> st1(int64_t i, int64_t j) { return {i + 1, j + 2}; }

static int check(const char* name,
                 void (*img)(int64_t*, int64_t, int64_t),
                 Maps fs,
                 std::array<int64_t, 2> ext,
                 std::array<int64_t, 2> shape,
                 std::array<int64_t, 2> a_lo,
                 std::array<int64_t, 2> a_hi) {
    int bad = 0;
    for (int64_t a0 = a_lo[0]; a0 < a_hi[0]; ++a0)
        for (int64_t a1 = a_lo[1]; a1 < a_hi[1]; ++a1) {
            int64_t r[8];
            img(r, a0, a1);
            std::set<int64_t> seen[2];
            for (int64_t i = a0; i < a0 + ext[0]; ++i)
                for (int64_t j = a1; j < a1 + ext[1]; ++j) {
                    for (auto f : fs) {
                        auto v = f(i, j);
                        for (int c = 0; c < 2; ++c)
                            if (v[c] >= 0 && v[c] < shape[c]) seen[c].insert(v[c]);
                    }
                }
            for (int c = 0; c < 2; ++c) {
                auto lo = r[c], hi = r[c] + r[2 + c];
                for (auto v : seen[c])
                    if ((v < lo || v >= hi) && ++bad <= 5)
                        std::printf("%s @(%lld, %lld) coord %d: %lld outside [%lld, %lld)\n", name, (long long)a0,
                                    (long long)a1, c, (long long)v, (long long)lo, (long long)hi);
                if ((r[2 + c] > r[4 + c] || r[4 + c] > shape[c]) && ++bad <= 5)
                    std::printf("%s @(%lld, %lld) coord %d: extent %lld beyond cap %lld or shape\n", name,
                                (long long)a0, (long long)a1, c, (long long)r[2 + c], (long long)r[4 + c]);
                if (r[6 + c])
                    for (auto v = lo; v < hi; ++v)
                        if (!seen[c].contains(v) && ++bad <= 5)
                            std::printf("%s @(%lld, %lld) coord %d: exact but %lld not reached\n", name, (long long)a0,
                                        (long long)a1, c, (long long)v);
            }
        }
    if (bad) std::printf("%s: %d failures\n", name, bad);
    return bad;
}

// Every point (i, j, k) of the «6, 8, 5» nest whose write (i + 2, j − 1) lies in the box at (y0, y1) of «2, 4».
static int check_pre() {
    int bad = 0;
    for (int64_t y0 = 0; y0 < 10; ++y0)
        for (int64_t y1 = 0; y1 < 10; ++y1) {
            int64_t r[10];
            pre_ws(r, y0, y1);
            for (int d = 0; d < 3; ++d)
                if (r[3 + d] > r[6 + d] && ++bad <= 5)
                    std::printf("pre @(%lld, %lld) dim %d: extent %lld beyond cap %lld\n", (long long)y0, (long long)y1,
                                d, (long long)r[3 + d], (long long)r[6 + d]);
            for (int64_t i = 0; i < 6; ++i)
                for (int64_t j = 0; j < 8; ++j)
                    for (int64_t k = 0; k < 5; ++k) {
                        int64_t x[3] = {i, j, k};
                        bool hits    = i + 2 >= y0 && i + 2 < y0 + 2 && j - 1 >= y1 && j - 1 < y1 + 4;
                        bool inside  = true;
                        for (int d = 0; d < 3; ++d)
                            inside = inside && x[d] >= r[d] && x[d] < r[d] + r[3 + d];
                        if (hits && !inside && ++bad <= 5)
                            std::printf("pre @(%lld, %lld): (%lld, %lld, %lld) writes the box but is outside\n",
                                        (long long)y0, (long long)y1, (long long)i, (long long)j, (long long)k);
                        if (r[9] && inside && !hits && ++bad <= 5)
                            std::printf("pre @(%lld, %lld): exact but (%lld, %lld, %lld) does not write the box\n",
                                        (long long)y0, (long long)y1, (long long)i, (long long)j, (long long)k);
                    }
        }
    if (bad) std::printf("pre: %d failures\n", bad);
    return bad;
}

int main() {
    int bad = 0;
    bad += check("win", img_win, {win}, {3, 3}, {64, 64}, {0, 0}, {20, 20});
    bad += check("resh", img_resh, {resh}, {1, 4}, {3, 8}, {0, 0}, {4, 3});
    bad += check("rdiv", img_rdiv, {rdiv}, {5, 4}, {8, 64}, {0, 0}, {20, 40});
    bad += check("trs", img_trs, {trs}, {2, 3}, {16, 16}, {1, 0}, {16, 16});
    bad += check("st", img_st, {st0, st1}, {3, 4}, {32, 32}, {1, 0}, {30, 28});
    bad += check_pre();
    return bad ? 1 : 0;
}
