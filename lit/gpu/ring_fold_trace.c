#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// The ABI of the residual program's only external operation.
void ring_event(uint64_t op, uint64_t x, uint64_t y, uint64_t z) {
    static const char *names[] = {"init", "expect", "fill", "poll", "release", "mma", "commit", "wait_group"};
    static const unsigned arity[] = {2, 2, 3, 2, 2, 2, 0, 1};
    if (op >= sizeof(arity) / sizeof(arity[0])) abort();
    fputs(names[op], stdout);
    uint64_t args[] = {x, y, z};
    for (unsigned i = 0; i < arity[op]; ++i) printf(" %" PRIu64, args[i]);
    putchar('\n');
}
