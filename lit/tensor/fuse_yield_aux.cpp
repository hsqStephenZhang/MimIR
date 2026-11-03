#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>

struct Pair {
    float *consumer, *outside;
};
extern "C" Pair exact(float*);
extern "C" Pair collision(float*);
extern "C" Pair exact_output(float*);
extern "C" Pair halo(float*);
extern "C" Pair one(float*);
extern "C" Pair permuted(float*);
extern "C" Pair rejected(float*);
extern "C" Pair multi(float*);
extern "C" Pair rejected_whole(float*);
static int64_t multiplies;
extern "C" void record_multiply(int64_t lanes) { multiplies += lanes; }

int main() {
    float input[20];
    for (int i = 0; i != 20; ++i)
        input[i] = i - 4.0f;
#ifdef WHOLE_BASELINE
    for (auto call : {rejected_whole}) {
#else
    for (auto call : {collision, exact, exact_output, one, rejected, halo, rejected_whole}) {
#endif
        multiplies         = 0;
        auto result        = call(input);
        auto expected_work = 10;
#ifdef BUDGET_ONE
        if (call == halo) expected_work = 17;
#endif
        if (multiplies != expected_work)
            std::fprintf(stderr, "%s: work=%lld expected=%d\n",
                         call == halo             ? "halo"
                         : call == exact          ? "exact"
                         : call == exact_output   ? "exact_output"
                         : call == rejected_whole ? "rejected_whole"
                         : call == collision      ? "collision"
                         : call == one            ? "one"
                                                  : "rejected",
                         (long long)multiplies, expected_work);
        assert(multiplies == expected_work);
        auto count = call == rejected ? 5 : 10;
        for (int i = 0; i != count; ++i)
            assert(result.consumer[i]
                   == (call == halo             ? 2 * (input[i] + (i == 9 ? 0 : input[i + 1]))
                       : call == rejected_whole ? 2 * input[i] + 10
                                                : 2 * input[i] + 1));
        for (int i = 0; i != 10; ++i)
            assert(result.outside[i] == 2 * input[i] + (call == exact_output ? 0 : 7));
        std::free(result.consumer);
        std::free(result.outside);
    }
#ifdef WHOLE_BASELINE
    return 0;
#endif
    multiplies = 0;
    auto three = multi(input);
    assert(multiplies == 10);
    for (int i = 0; i != 10; ++i) {
        assert(three.consumer[i] == 4 * input[i] + 8);
        assert(three.outside[i] == 2 * input[i] + 7);
    }
    std::free(three.consumer);
    std::free(three.outside);
    float matrix[30];
    for (int i = 0; i != 30; ++i)
        matrix[i] = i - 11.0f;
    multiplies   = 0;
    auto columns = permuted(matrix);
    assert(multiplies == 30);
    for (int j = 0; j != 10; ++j) {
        if (columns.consumer[j] != 2 * (matrix[j] + matrix[10 + j] + matrix[20 + j]))
            std::fprintf(stderr, "column %d: %g expected %g\n", j, columns.consumer[j],
                         2 * (matrix[j] + matrix[10 + j] + matrix[20 + j]));
        assert(columns.consumer[j] == 2 * (matrix[j] + matrix[10 + j] + matrix[20 + j]));
        for (int i = 0; i != 3; ++i)
            assert(columns.outside[i * 10 + j] == 2 * matrix[i * 10 + j] + 7);
    }
    std::free(columns.consumer);
    std::free(columns.outside);
}
