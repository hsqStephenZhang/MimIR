#include <stdint.h>
#include <stdlib.h>

extern int32_t* make_vector(uint64_t n);
extern int32_t* make_matrix(uint64_t n);
extern int32_t update_last(uint64_t n, int32_t* a);
extern int32_t* vector_one(void);
extern void* vector_zero(void);

int main(void) {
    const uint64_t sizes[] = {0, 1, 2, 7, 17};
    for (unsigned k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        uint64_t n      = sizes[k];
        int32_t* vector = make_vector(n);
        int32_t* matrix = make_matrix(n);
        for (uint64_t i = 0; i < n; ++i)
            if (vector[i] != 7) return 1;
        for (uint64_t i = 0; i < 3 * n; ++i)
            if (matrix[i] != 7) return 2;
        if (n) {
            if (update_last(n, matrix) != 9) return 3;
            for (uint64_t i = 0; i < 3 * n; ++i)
                if (matrix[i] != (i + 1 == 3 * n ? 9 : 7)) return 4;
        }
        free(vector);
        free(matrix);
    }
    int32_t* one = vector_one();
    if (*one != 7) return 5;
    free(one);
    free(vector_zero());
    return 0;
}
