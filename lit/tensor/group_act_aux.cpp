#include <cstdint>

extern "C" int32_t* counted(int32_t*, int32_t*);

// x[i, j, k] = 12 i + 3 j + k, so C[j] = 558 + 54 j, and the action runs once for each of the 4 columns.
int main() {
    int32_t x[6 * 4 * 3];
    for (int i = 0; i != 6 * 4 * 3; ++i)
        x[i] = i;
    int32_t n = 0;
    auto c    = counted(x, &n);
    int bad   = n != 4;
    for (int j = 0; j != 4; ++j)
        bad += c[j] != 558 + 54 * j;
    return bad;
}
