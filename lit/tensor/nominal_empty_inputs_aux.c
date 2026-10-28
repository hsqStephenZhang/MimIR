#include <stdint.h>
#include <stdlib.h>

extern int32_t* pad_empty(void* a);
extern int32_t* pad_dynamic(uint64_t n, int32_t* a);
extern int32_t* concat_empty(void* a, int32_t* b);
extern void* concat_all_empty(void* a);
extern void* broadcast_empty(uint64_t n, void* a);

int main(void) {
    int32_t input[] = {2, 3, 5, 11};
    int32_t* padded = pad_empty(NULL);
    if (padded[0] != 7 || padded[1] != 7) return 1;
    free(padded);
    int32_t* joined = concat_empty(NULL, input);
    for (unsigned i = 0; i < 8; ++i)
        if (joined[i] != input[i % 4]) return 2;
    free(joined);
    free(concat_all_empty(NULL));
    for (uint64_t n = 0; n <= 4; ++n) {
        free(broadcast_empty(n, NULL));
        padded = pad_dynamic(n, n ? input : NULL);
        if (padded[0] != 7 || padded[n + 1] != 7) return 3;
        for (uint64_t i = 0; i < n; ++i)
            if (padded[i + 1] != input[i]) return 4;
        free(padded);
    }
    return 0;
}
