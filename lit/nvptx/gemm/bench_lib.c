#include <stdint.h>
#include <math.h>
#include <stdio.h>

static int is_close_f32(float actual, float expected, double rtol, double atol) {
    if (actual == expected) return 1;
    if (!isfinite(actual) || !isfinite(expected)) return 0;

    double diff = fabs((double)actual - (double)expected);
    double tol = atol + rtol * fabs((double)expected);
    return diff <= tol;
}

int32_t assert_close_scalar(float actual, float expected, double rtol, double atol) {
    if (is_close_f32(actual, expected, rtol, atol)) return 0;

    double diff = fabs((double)actual - (double)expected);
    double tol = atol + rtol * fabs((double)expected);
    fprintf(stderr,
            "assert_close_scalar failed: actual=%g expected=%g diff=%g tol=%g "
            "(rtol=%g, atol=%g)\n",
            actual, expected, diff, tol, rtol, atol);
    return 1;
}

int32_t assert_close(int32_t n, const float* actual, const float* expected, double rtol, double atol) {
    for (int32_t i = 0; i < n; ++i) {
        if (is_close_f32(actual[i], expected[i], rtol, atol)) continue;

        double diff = fabs((double)actual[i] - (double)expected[i]);
        double tol = atol + rtol * fabs((double)expected[i]);
        fprintf(stderr,
                "assert_close failed at index %d/%d: actual=%g expected=%g diff=%g tol=%g "
                "(rtol=%g, atol=%g)\n",
                i, n, actual[i], expected[i], diff, tol, rtol, atol);
        return 1;
    }
    return 0;
}

void bench_print_gemm_header(int32_t m, int32_t n, int32_t k) {
    printf("Matrix Dimensions: M=%d, N=%d, K=%d\n", m, n, k);
    printf("\n--- Performance Results ---\n");
}

void bench_print_gemm_result(int32_t variant,
                             double avg_ms,
                             double avg_ns,
                             double total_ms,
                             int32_t iterations,
                             double gflops) {
    (void)avg_ns;
    (void)total_ms;
    (void)iterations;

    const char* name = "Unknown GEMM";
    if (variant == 0) name = "Naive GEMM";
    if (variant == 1) name = "Tiled GEMM (16x16)";
    if (variant == 2) name = "Double Buffered GEMM";
    if (variant == 3) name = "TVM-style GEMM";
    if (variant == 4) name = "Pipelined TVM-style GEMM";

    printf("%-30s : %8.3f ms | %8.3f GFLOPS\n", name, avg_ms, gflops);
}

void bench_print_gemm_verify(int32_t variant, int32_t status) {
    const char* name = "Unknown GEMM";
    if (variant == 0) name = "Naive GEMM";
    if (variant == 1) name = "Tiled GEMM (16x16)";
    if (variant == 2) name = "Double Buffered GEMM";
    if (variant == 3) name = "TVM-style GEMM";
    if (variant == 4) name = "Pipelined TVM-style GEMM";

    if (status == 0)
        printf("  %s Verification successful!\n", name);
    else
        printf("  %s Verification failed!\n", name);
}
