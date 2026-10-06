#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

#include "guarded_input.h"
extern "C" {
int32_t* masked_rows(int32_t*);
int32_t* nested_padding(int32_t*);
int32_t* nested_reduction(int32_t*);
int32_t* masked_epilogue(int32_t*);
int32_t* empty_reduction(int32_t*);
int32_t* masked_gemm(int32_t*, int32_t*);
int32_t* masked_bias(int32_t*, int32_t*);
}
int main() {
    std::vector<int32_t> input(5 * 7 * 13);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = int32_t(i % 41) - 20;
    GuardedInput guarded_input(input);
    for (auto f : {masked_rows, nested_padding, nested_reduction, masked_epilogue, empty_reduction}) {
        for (int rep = 0; rep < 10; ++rep) {
            auto out = f(guarded_input.data());
            for (size_t i = 0; i < 35; ++i) {
                int32_t expected = f == empty_reduction ? 37 : f == masked_epilogue ? 1000 : 0;
                if (f != empty_reduction)
                    for (size_t k = 0; k < 13; ++k)
                        expected += input[i * 13 + k];
                if (out[i] != expected) {
                    std::fprintf(stderr, "masked output[%zu] = %d, expected %d\n", i, out[i], expected);
                    std::free(out);
                    return 1;
                }
            }
            std::free(out);
        }
    }
    std::vector<int32_t> bias(35);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = int32_t(i) * 3;
    GuardedInput guarded_bias(bias);
    auto biased = masked_bias(guarded_input.data(), guarded_bias.data());
    for (size_t i = 0; i < 35; ++i) {
        auto expected = bias[i];
        for (size_t k = 0; k < 13; ++k)
            expected += input[i * 13 + k];
        if (biased[i] != expected) {
            std::fprintf(stderr, "masked bias[%zu] = %d, expected %d\n", i, biased[i], expected);
            std::free(biased);
            return 1;
        }
    }
    std::free(biased);
    std::vector<int32_t> a(5 * 13), b(13 * 19);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = int32_t(i % 9) - 4;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = int32_t(i % 11) - 5;
    GuardedInput guarded_a(a), guarded_b(b);
    auto out = masked_gemm(guarded_a.data(), guarded_b.data());
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 19; ++j) {
            int32_t expected = 0;
            for (size_t k = 0; k < 13; ++k)
                expected += a[i * 13 + k] * b[k * 19 + j];
            if (out[i * 19 + j] != expected) {
                std::fprintf(stderr, "masked GEMM[%zu,%zu] = %d, expected %d\n", i, j, out[i * 19 + j], expected);
                std::free(out);
                return 1;
            }
        }
    std::free(out);
}
