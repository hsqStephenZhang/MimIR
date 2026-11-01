#include <cstdint>
#include <cstdio>

extern "C" int32_t narrow(int64_t*, int64_t*);

int main() {
    int64_t src[3] = {7, -4, 11}, dst[3] = {};
    if (narrow(src, dst) != 0) return 1;
    for (int i = 0; i != 3; ++i) {
        if (dst[i] != src[i] * 3) {
            std::fprintf(stderr, "cell %d: expected %lld, got %lld\n", i, static_cast<long long>(src[i] * 3),
                         static_cast<long long>(dst[i]));
            return 1;
        }
    }
}
