#include <stdbool.h>
#include <stdint.h>

extern int32_t local_tensor(uint64_t n, int32_t x);
extern int32_t branch_tensor(uint64_t n, int32_t x, bool active);

int main(void) {
    const uint64_t sizes[] = {0, 1, 2, 7, 17};
    for (unsigned k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        if (local_tensor(sizes[k], 11) != 11 + sizes[k]) return 1;
        if (branch_tensor(sizes[k], 11, false) != 11) return 2;
        if (branch_tensor(sizes[k], 11, true) != 11 + sizes[k]) return 3;
    }
    return 0;
}
