#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <initializer_list>

extern "C" float* shared(float*);
extern "C" float* nested(float*);
extern "C" float* boxes(float*);
extern "C" float* distinct(float*);

static uint64_t multiplications = 0;
extern "C" void record_multiply(uint64_t lanes) { multiplications += lanes; }

int main() {
    float x[8];
    for (int repeat = 0; repeat != 5; ++repeat) {
        for (int i = 0; i != 8; ++i)
            x[i] = float(i + repeat * 8);
        for (auto function : {shared, nested, distinct}) {
            multiplications = 0;
            auto result     = function(x);
#ifdef REALIZATION_COUNT
            if (function != distinct) assert(multiplications == 8);
#endif
            for (int i = 0; i != 8; ++i) {
                auto factor = function == distinct ? 3.125f + 3.125f * 3.125f : 6.25f;
                assert(result[i] == factor * x[i]);
            }
            std::free(result);
        }
        multiplications = 0;
        auto result     = boxes(x);
        for (int i = 0; i != 4; ++i)
            assert(result[i] == 3.125f * (x[i] + x[i + 4]));
        std::free(result);
    }
}
