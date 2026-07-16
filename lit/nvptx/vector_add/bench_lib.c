#include <stdio.h>
#include <stdint.h>

void bench_print_vector_add(int32_t n_elements,
                            double avg_ms,
                            double avg_ns,
                            double total_ms,
                            int32_t iterations,
                            double gflops,
                            double gb_per_sec) {
    printf("==================================================\n");
    printf("GPU Vector Add Performance (N = %d elements)\n", n_elements);
    printf("Execution Time : %.4f ms/iter (%.0f ns/iter)\n", avg_ms, avg_ns);
    printf("Total Time     : %.4f ms (%d iterations)\n", total_ms, iterations);
    printf("Throughput     : %.4f GFLOPS\n", gflops);
    printf("Bandwidth      : %.4f GB/s\n", gb_per_sec);
    printf("==================================================\n");
}
