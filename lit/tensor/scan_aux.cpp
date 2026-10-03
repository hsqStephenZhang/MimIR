// Driver for scan_exec.mim: checks the running sum `tensor.scan` computes along the leading axis.
// Exit code: 0 if every output matches, 1 otherwise.

#include <cstdint>
#include <cstdio>

extern "C" {
// `«s; I32»` tensors lower to plain pointers at the C ABI (see fc_aux.cpp).
int32_t* prefix_sum(int32_t* input);
}

int main() {
    constexpr size_t N = 5, C = 3;
    int32_t in[N * C];
    for (size_t i = 0; i < N * C; ++i)
        in[i] = int32_t(i % 7) - 3;

    int32_t* out = prefix_sum(in);

    int bad = 0;
    int32_t acc[C] = {0, 0, 0};
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c) {
            acc[c] += in[n * C + c];
            if (out[n * C + c] != acc[c]) {
                if (++bad <= 5) std::printf("prefix_sum[%zu,%zu]: got %d want %d\n", n, c, out[n * C + c], acc[c]);
            }
        }
    if (bad) std::printf("prefix_sum: %d mismatches\n", bad);
    return bad ? 1 : 0;
}
