// Exercise the runtime directly: independent submissions, nested calls, changing task sizes,
// and empty/very large ranges. Each process starts with concurrent first-use initialization.
#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#    include <pthread.h>
#    include <sched.h>
#endif

void mim_rt_parallel_for(int64_t n, void (*body)(void*, int64_t, int64_t), void* env);

enum { extent = 257, callers = 4, repetitions = 500 };
static atomic_int errors;

static void fill(void* env, int64_t lo, int64_t hi) {
    atomic_int* hits = env;
    for (int64_t i = lo; i < hi; ++i)
        atomic_fetch_add(&hits[i], 1);
}

static void check_sizes(void) {
    atomic_int hits[extent];
    const int sizes[] = {2, 8, 1, 0, -1, 3, extent, 7};
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        for (unsigned s = 0; s < sizeof sizes / sizeof *sizes; ++s) {
            for (int i = 0; i < extent; ++i)
                atomic_init(&hits[i], 0);
            int n = sizes[s];
            mim_rt_parallel_for(n, fill, hits);
            for (int i = 0; i < extent; ++i)
                if (atomic_load(&hits[i]) != (i < n)) atomic_fetch_add(&errors, 1);
        }
    }
}

#if !defined(_WIN32)
static atomic_int ready, start;
static void* caller(void* unused) {
    (void)unused;
    atomic_fetch_add(&ready, 1);
    while (!atomic_load(&start))
        sched_yield();
    check_sizes();
    return NULL;
}
#endif

static void nested(void* env, int64_t lo, int64_t hi) {
    atomic_int* hits = env;
    for (int64_t i = lo; i < hi; ++i)
        mim_rt_parallel_for(17, fill, &hits[i * 17]);
}

struct ranges {
    atomic_int count;
    int64_t lo[extent], hi[extent];
};

static void record_range(void* env, int64_t lo, int64_t hi) {
    struct ranges* ranges = env;
    int index             = atomic_fetch_add(&ranges->count, 1);
    if (index >= extent) abort();
    ranges->lo[index] = lo;
    ranges->hi[index] = hi;
}

int main(void) {
#if !defined(_WIN32)
    pthread_t threads[callers];
    for (int i = 0; i < callers; ++i)
        if (pthread_create(&threads[i], NULL, caller, NULL)) return 2;
    while (atomic_load(&ready) != callers)
        sched_yield();
    atomic_store(&start, 1);
    for (int i = 0; i < callers; ++i)
        if (pthread_join(threads[i], NULL)) return 2;
#else
    check_sizes();
#endif

    atomic_int hits[32 * 17];
    for (int i = 0; i < 32 * 17; ++i)
        atomic_init(&hits[i], 0);
    mim_rt_parallel_for(32, nested, hits);
    for (int i = 0; i < 32 * 17; ++i)
        if (atomic_load(&hits[i]) != 1) atomic_fetch_add(&errors, 1);

    struct ranges ranges = {0};
    mim_rt_parallel_for(INT64_MAX, record_range, &ranges);
    int count   = atomic_load(&ranges.count);
    int64_t end = 0;
    for (int i = 0; i < count; ++i) {
        int next = -1;
        for (int j = 0; j < count; ++j)
            if (ranges.lo[j] == end) next = j;
        if (next < 0 || ranges.hi[next] <= end) return 1;
        end = ranges.hi[next];
    }
    if (!count || end != INT64_MAX) return 1;
    int bad = atomic_load(&errors);
    if (bad) fprintf(stderr, "parallel_for: %d incorrect iteration counts\n", bad);
    return bad ? 1 : 0;
}
