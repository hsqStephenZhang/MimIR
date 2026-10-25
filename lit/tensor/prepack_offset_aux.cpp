#include <cstdint>
#include <initializer_list>

extern "C" int32_t* shifted(int32_t*, uint64_t);

// x[i, j] = 3 i + j, so P[i] = 9 (i + off) + 3.
int main() {
    int32_t x[8 * 3];
    for (int i = 0; i != 8 * 3; ++i)
        x[i] = i;
    int bad = 0;
    for (uint64_t off : {0, 2}) {
        auto p = shifted(x, off);
        for (int i = 0; i != 4; ++i)
            bad += p[i] != 9 * (i + int(off)) + 3;
    }
    return bad;
}
