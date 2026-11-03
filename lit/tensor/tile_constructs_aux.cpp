#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* tiled_tail(int64_t*);
extern "C" int64_t* tiled_one(int64_t*);
extern "C" int64_t* tiled_full(int64_t*);
extern "C" int64_t* tiled_scalar(int64_t*);
extern "C" int64_t* packed(int64_t*);
extern "C" int64_t* packed_both(int64_t*);
extern "C" int64_t* packed_one(int64_t*);
extern "C" int64_t* packed_box(int64_t*);
extern "C" int64_t* packed_box_one(int64_t*);
extern "C" int64_t* tensor_fold(int64_t*);
extern "C" int64_t* fold_shared_read(int64_t*);
extern "C" int64_t* fold_extra_read(int64_t*);
extern "C" uint64_t ordered();
extern "C" uint64_t no_blocks();
extern "C" int64_t* seeded_zero();
extern "C" int64_t* seeded_one();
extern "C" int64_t* seeded_three();

int main() {
    int64_t x[35];
    for (int i = 0; i != 35; ++i)
        x[i] = i * 7 - 9;
    auto tile = tiled_tail(x);
    auto one  = tiled_one(x);
    auto full       = tiled_full(x);
    auto scalar     = tiled_scalar(x);
    auto p          = packed(x);
    auto q          = packed_both(x);
    auto s          = packed_one(x);
    auto box        = packed_box(x);
    auto box_one    = packed_box_one(x);
    auto f          = tensor_fold(x);
    auto shared     = fold_shared_read(x);
    auto extra      = fold_extra_read(x);
    bool ok         = ordered() == 1024 && no_blocks() == 7;
    auto seed_zero  = seeded_zero();
    auto seed_one   = seeded_one();
    auto seed_three = seeded_three();
    for (int i = 0; i != 3; ++i) {
        if (seed_zero[i] != 3 || seed_one[i] != 30 || seed_three[i] != 3024)
            std::fprintf(stderr, "seed cell %d: expected 3/30/3024, got %lld/%lld/%lld\n", i,
                         static_cast<long long>(seed_zero[i]), static_cast<long long>(seed_one[i]),
                         static_cast<long long>(seed_three[i]));
        ok &= seed_zero[i] == 3 && seed_one[i] == 30 && seed_three[i] == 3024;
    }
    for (int i = 0; i != 35; ++i)
        ok &= tile[i] == x[i];
    for (int i = 0; i != 5; ++i)
        ok &= one[i] == x[i];
    for (int i = 0; i != 4; ++i)
        for (int j = 0; j != 6; ++j)
            ok &= full[i * 6 + j] == x[i * 7 + j];
    ok &= scalar[0] == x[0];
    std::free(scalar);
    for (int j = 0; j != 3; ++j)
        for (int i = 0; i != 5; ++i)
            for (int lane = 0; lane != 3; ++lane) {
                auto col = j * 3 + lane;
                ok &= p[(j * 5 + i) * 3 + lane] == (col < 7 ? x[i * 7 + col] : -99);
            }
    for (int j = 0; j != 3; ++j)
        for (int i = 0; i != 3; ++i)
            for (int jl = 0; jl != 3; ++jl)
                for (int il = 0; il != 2; ++il) {
                    auto row = i * 2 + il, col = j * 3 + jl;
                    ok &= q[((j * 3 + i) * 3 + jl) * 2 + il] == (row < 5 && col < 7 ? x[row * 7 + col] : -99);
                }
    for (int i = 0; i != 6; ++i)
        ok &= s[i] == (i < 5 ? x[i] : -99);
    for (int j = 0; j != 2; ++j)
        for (int i = 0; i != 2; ++i)
            for (int lane = 0; lane != 2; ++lane) {
                auto col = j * 2 + lane;
                ok &= box[(j * 2 + i) * 2 + lane] == (col < 3 ? x[(1 + i * 2) * 7 + 1 + col * 2] : -99);
            }
    for (int i = 0; i != 4; ++i)
        ok &= box_one[i] == (i < 3 ? x[2 * 7 + 1 + i * 2] : -99);
    for (int i = 0; i != 3; ++i)
        ok &= f[i] == x[i] + 6 && shared[i] == x[0] + 6 && extra[i] == x[i] * 8;
    std::free(tile);
    std::free(one);
    std::free(full);
    std::free(p);
    std::free(q);
    std::free(s);
    std::free(box);
    std::free(box_one);
    std::free(f);
    std::free(shared);
    std::free(extra);
    std::free(seed_zero);
    std::free(seed_one);
    std::free(seed_three);
    if (!ok) std::fprintf(stderr, "tiled, stage or ordered-fold reference mismatch\n");
    return ok ? 0 : 1;
}
