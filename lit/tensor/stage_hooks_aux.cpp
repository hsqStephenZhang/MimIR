#include <cstdint>

extern "C" {
int32_t* moved(int32_t*, int32_t*);
int32_t* after(int32_t*, int32_t*);
int32_t* copied(int32_t*, int32_t*);
}

// x[i, j, k] = 12 i + 3 j + k, so Σ_k x[i, j, k] = 36 i + 9 j + 3, and moved's C[j] = 558 + 54 j.
int main() {
    int32_t x[6 * 4 * 3];
    for (int i = 0; i != 6 * 4 * 3; ++i)
        x[i] = i;
    int bad = 0;

    int32_t n = 0;
    auto c    = moved(x, &n);
    bad += n != 6 * (1 + 2 + 3 + 4);
    for (int j = 0; j != 4; ++j)
        bad += c[j] != 558 + 54 * j;

    n = 0;
    c = after(x, &n);
    bad += n != 6;
    for (int i = 0; i != 6 * 4; ++i)
        bad += c[i] != 36 * (i / 4) + 9 * (i % 4) + 4;

    n = 0;
    c = copied(x, &n);
    bad += n != 6;
    for (int i = 0; i != 6 * 4; ++i)
        bad += c[i] != i + 1;
    return bad;
}
