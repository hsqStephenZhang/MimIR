#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
int32_t* direct(int32_t*);
int32_t* explicit_domain(int32_t*);
int32_t* permuted(int32_t*);
int32_t* tiled(int32_t*);
int32_t* pointwise(int32_t*);
int32_t* rescheduled(int32_t*);
int32_t* epilogue(int32_t*);
int32_t* three_axes(int32_t*);
int32_t* reduction_axes(int32_t*);
int32_t* empty_outer(int32_t*);
int32_t* empty_inner(int32_t*);
}

int main() {
    std::vector<int32_t> input(6 * 10 * 7);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = int32_t(i % 41) - 20;
    for (auto f : {direct, permuted, tiled, rescheduled, epilogue, explicit_domain}) {
        for (int rep = 0; rep < 20; ++rep) {
            auto out = f(input.data());
            for (size_t p = 0; p < 60; ++p) {
                int32_t want = f == epilogue ? 1000 : 0;
                for (size_t k = 0; k < 7; ++k)
                    want += input[p * 7 + k];
                if (out[p] != want) {
                    std::fprintf(stderr, "fused output[%zu] = %d, expected %d\n", p, out[p], want);
                    std::free(out);
                    return 1;
                }
            }
            std::free(out);
        }
    }
    for (auto f : {three_axes, reduction_axes}) {
        const size_t cells = f == three_axes ? 30 : 60;
        const size_t taps  = f == three_axes ? 7 : 15;
        std::vector<int32_t> values(cells * taps);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = int32_t(i % 37) - 18;
        auto out = f(values.data());
        for (size_t p = 0; p < cells; ++p) {
            int32_t want = 0;
            for (size_t k = 0; k < taps; ++k)
                want += values[p * taps + k];
            if (out[p] != want) {
                std::fprintf(stderr, "multi-axis output[%zu] = %d, expected %d\n", p, out[p], want);
                std::free(out);
                return 1;
            }
        }
        std::free(out);
    }
    auto copied = pointwise(input.data());
    for (size_t p = 0; p < 60; ++p) {
        if (copied[p] != input[p]) {
            std::fprintf(stderr, "pointwise output[%zu] = %d, expected %d\n", p, copied[p], input[p]);
            std::free(copied);
            return 1;
        }
    }
    std::free(copied);
    for (auto f : {empty_outer, empty_inner})
        std::free(f(input.data()));
}
