#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int32_t *a, *b;
} pair;
extern int32_t* make_array(uint64_t n);
extern pair make_pair(uint64_t n);
extern int32_t* capture(uint64_t n, int32_t* a);
extern int32_t* sum_empty(void);
extern void* empty_map(void);
extern int32_t read_marker(uint64_t n, int32_t* a, int32_t marker);
extern int32_t* static_one(void);
extern int32_t* make_scalar(void);
extern void* static_zero(void);
extern int32_t* call_one(void);
extern int32_t* closed_update(void);
extern void* call_zero(void);

int main(void) {
    uint64_t sizes[] = {0, 1, 2, 7, 17};
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        uint64_t n = sizes[k];
        int32_t* a = make_array(n);
        for (uint64_t i = 0; i < n; ++i)
            if (a[i] != 2 * i + 3) return 10;
        free(a);
        int32_t* packed = malloc(n * sizeof(*packed));
        for (uint64_t i = 0; i < n; ++i)
            packed[i] = 3 * i + 1;
        for (uint64_t i = 0; i < n; ++i)
            if (packed[i] != 3 * i + 1) return 16;
        int32_t* incremented = capture(n, packed);
        for (uint64_t i = 0; i < n; ++i)
            if (incremented[i] != 3 * i + 2 || packed[i] != 3 * i + 1) return 17;
        if (n && read_marker(n, packed, 41) != 42) return 18;
        free(incremented);
        free(packed);
        if (n) {
            pair p = make_pair(n);
            if (p.a == p.b) return 11;
            for (uint64_t i = 0; i < n; ++i) {
                if (p.a[i] != 2 * i + 3) return 12;
                if (p.b[i] != (i == n - 1 ? 9 : 2 * i + 3)) return 13;
            }
            free(p.a);
            free(p.b);
        }
        printf("runtime n=%lu: PASS\n", (unsigned long)n);
    }
    int32_t *one = static_one(), *zero_rank = make_scalar();
    if (*one != 5 || *zero_rank != 42) return 14;
    free(one);
    free(zero_rank);
    free(static_zero());
    int32_t* adapted = call_one();
    if (*adapted != 3) return 15;
    free(adapted);
    free(call_zero());
    int32_t* reduced = sum_empty();
    if (reduced[0] != 5 || reduced[1] != 5 || reduced[2] != 5) return 19;
    free(reduced);
    free(empty_map());
    int32_t* updated = closed_update();
    if (updated[0] != 42 || updated[1] != 2 || updated[2] != 3 || updated[3] != 4) return 20;
    free(updated);
    puts("static extent-0/1 calls and rank-0 tensor: PASS");
    return 0;
}
