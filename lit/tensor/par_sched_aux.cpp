// Driver for par_sched_exec.mim: checks sum_par's row sums against a scalar reference.
// Exit code: 0 if every row matches, 1 otherwise.

#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" {
int32_t* sum_par(int32_t* input);
}

int main() {
    const int P = 32, K = 16;
    std::vector<int32_t> input(P * K);
    for (int p = 0; p < P; ++p)
        for (int k = 0; k < K; ++k) input[p * K + k] = p * K + k + 1;

    int32_t* out = sum_par(input.data());

    int bad = 0;
    for (int p = 0; p < P; ++p) {
        long want = 0;
        for (int k = 0; k < K; ++k) want += input[p * K + k];
        if (out[p] != want) {
            if (bad < 5) std::printf("sum_par[%d] = %d, want %ld\n", p, out[p], want);
            ++bad;
        }
    }
    return bad ? 1 : 0;
}
