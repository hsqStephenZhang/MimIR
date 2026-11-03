#include <cassert>
#include <cstdlib>

extern "C" float* plain(float*, float*);
extern "C" float* matrix(float*);

int main() {
    float y[10], z[10], input[20];
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (int i = 0; i != 10; ++i) {
            y[i] = float(3 * i - 11 + repeat);
            z[i] = float(17 - 2 * i - repeat);
        }
        for (int i = 0; i != 20; ++i)
            input[i] = float(i * 7 - 19 + repeat);
        auto sum  = plain(y, z);
        auto grid = matrix(input);
        for (int i = 0; i != 10; ++i)
            assert(sum[i] == y[i] + z[i]);
        for (int i = 0; i != 20; ++i)
            assert(grid[i] == input[i] + 1);
        std::free(sum);
        std::free(grid);
    }
}
