#include <cassert>
#include <cstdlib>

struct Pair {
    float *first, *second;
};

extern "C" Pair collision(float*);

int main() {
    float input[20];
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (int i = 0; i != 20; ++i)
            input[i] = float(i * 3 - 11 + repeat);
        auto result = collision(input);
        for (int i = 0; i != 10; ++i) {
            assert(result.first[i] == input[i]);
            assert(result.second[i] == 2 * input[i]);
        }
        std::free(result.first);
        std::free(result.second);
    }
}
