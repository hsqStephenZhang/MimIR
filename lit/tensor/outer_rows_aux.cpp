#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <array>

struct Pair {
    float *producer, *consumer;
};
extern "C" Pair accepted(float*, float*, float*);
extern "C" Pair whole(float*, float*, float*);
static int64_t work;
extern "C" void record_multiply(int64_t lanes) { work += lanes; }
extern "C" void record_center(int64_t lanes) { work += lanes; }

int main() {
    float input[33 * 768], weights[768 * 8];
    for (int i = 0; i != 33 * 768; ++i)
        input[i] = (i % 7 - 3) * .001f;
    for (int i = 0; i != 768 * 8; ++i)
        weights[i] = (i % 11 - 5) * .01f;
    for (int rows : std::array<int, 2>{33, 7}) {
        work        = 0;
        auto result = (rows == 33 ? accepted : whole)(input, weights, weights);
#ifdef COUNT_CENTER
        assert(work == rows * 768);
#else
        assert(work == rows * 768 * 8);
#endif
        for (int i = 0; i != rows; ++i) {
            float total = 0;
            for (int k = 0; k != 768; ++k)
                total += input[i * 768 + k];
            for (int k = 0; k != 768; ++k)
                assert(std::abs(result.producer[i * 768 + k] - (2 * input[i * 768 + k] - total)) < .00001f);
            for (int j = 0; j != 8; ++j) {
                float expected = weights[j];
                for (int k = 0; k != 768; ++k)
                    expected += (input[i * 768 + k] - total) * weights[k * 8 + j];
                assert(std::abs(result.consumer[i * 8 + j] - expected) < .00001f);
            }
        }
        std::free(result.producer);
        std::free(result.consumer);
    }
}
