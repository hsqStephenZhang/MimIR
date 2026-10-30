#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <array>
#include <utility>

extern "C" uint64_t* qkv(uint64_t*, uint64_t*, uint64_t*, uint64_t*);

int main() {
    std::array<uint64_t, 6> x;
    std::array<uint64_t, 12> q;
    std::array<uint64_t, 15> k;
    std::array<uint64_t, 18> v;
    for (uint64_t seed = 0; seed != 5; ++seed) {
        for (size_t i = 0; i != x.size(); ++i)
            x[i] = (i * 7 + seed) % 11;
        for (size_t i = 0; i != q.size(); ++i)
            q[i] = (i * 3 + seed) % 13;
        for (size_t i = 0; i != k.size(); ++i)
            k[i] = (i * 5 + seed) % 17;
        for (size_t i = 0; i != v.size(); ++i)
            v[i] = (i * 2 + seed) % 19;
        if (seed == 4) {
            for (auto& value : x)
                value = UINT64_MAX - value;
            for (auto& value : q)
                value = UINT64_MAX - value;
            for (auto& value : k)
                value = UINT64_MAX - value;
            for (auto& value : v)
                value = UINT64_MAX - value;
        }
        auto out      = qkv(x.data(), q.data(), k.data(), v.data());
        size_t offset = 0;
        for (auto [width, weight] : {
                 std::pair{size_t(4), q.data()},
                 {size_t(5), k.data()},
                 {size_t(6), v.data()}
        }) {
            for (size_t row = 0; row != 2; ++row)
                for (size_t col = 0; col != width; ++col) {
                    uint64_t expected = 0;
                    for (size_t c = 0; c != 3; ++c)
                        expected += x[row * 3 + c] * weight[c * width + col];
                    if (out[row * 15 + offset + col] != expected) {
                        std::fprintf(stderr, "qkv seed %llu row %zu col %zu: got %llu, expected %llu\n",
                                     (unsigned long long)seed, row, offset + col,
                                     (unsigned long long)out[row * 15 + offset + col], (unsigned long long)expected);
                        std::free(out);
                        return 1;
                    }
                }
            offset += width;
        }
        std::free(out);
    }
}
