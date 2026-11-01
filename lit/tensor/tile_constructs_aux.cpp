#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* tiled_tail(int64_t*);
extern "C" int64_t* tiled_one(int64_t*);
extern "C" void tiled_empty();
extern "C" int64_t* tiled_full(int64_t*);
extern "C" int64_t tiled_scalar(int64_t);
extern "C" int64_t* packed(int64_t*);
extern "C" int64_t* packed_both(int64_t*);
extern "C" int64_t* packed_one(int64_t*);
extern "C" int64_t* tensor_fold(int64_t*);
extern "C" int64_t* fold_shared_read(int64_t*);
extern "C" int64_t* fold_extra_read(int64_t*);
extern "C" uint64_t ordered();
extern "C" uint64_t no_blocks();

int main() {
    int64_t x[35];
    for (int i = 0; i != 35; ++i)
        x[i] = i * 7 - 9;
    auto tile = tiled_tail(x);
    auto one  = tiled_one(x);
    tiled_empty();
    auto full   = tiled_full(x);
    auto scalar = tiled_scalar(x[0]);
    auto p      = packed(x);
    auto q      = packed_both(x);
    auto s      = packed_one(x);
    auto f      = tensor_fold(x);
    auto shared = fold_shared_read(x);
    auto extra  = fold_extra_read(x);
    bool ok     = ordered() == 1024 && no_blocks() == 7;
    for (int i = 0; i != 35; ++i)
        ok &= tile[i] == x[i];
    for (int i = 0; i != 5; ++i)
        ok &= one[i] == x[i];
    for (int i = 0; i != 4; ++i)
        for (int j = 0; j != 6; ++j)
            ok &= full[i * 6 + j] == x[i * 7 + j];
    ok &= scalar == x[0];
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
    for (int i = 0; i != 3; ++i)
        ok &= f[i] == x[i] + 6 && shared[i] == x[0] + 6 && extra[i] == x[i] * 8;
    std::free(tile);
    std::free(one);
    std::free(full);
    std::free(p);
    std::free(q);
    std::free(s);
    std::free(f);
    std::free(shared);
    std::free(extra);
    if (!ok) std::fprintf(stderr, "tiled, stage or ordered-fold reference mismatch\n");
    return ok ? 0 : 1;
}
