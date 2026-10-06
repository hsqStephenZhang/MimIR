// Runtime wrappers for the MimIR `ll` backend.
//
// The functions in this file are implemented in C, compiled to textual LLVM IR by `clang`
// at build time (see `add_mim_runtime` in `cmake/Mim.cmake`), and then either embedded into
// or linked with the module emitted by the `ll` backend (selected via `-X ll:rt=embed|extern`).
//
// This lets the emitter offload complex or platform-dependent lowerings to `clang` instead of
// hand-writing raw LLVM IR; see issue #486.
// Wrappers must expose a flat, scalar ABI (no C aggregates across the boundary) so that the
// emitter can lower a Mim intrinsic to a single `call`.

#include <setjmp.h>
#include <stdint.h>

/// Size in bytes of a `jmp_buf`, used by `clos.alloc_jmpbuf` to reserve stack space.
/// The size is platform- and libc-dependent, so we let the C compiler compute it rather than
/// hard-coding it in the backend.
int64_t mim_jmpbuf_size(void) { return (int64_t)sizeof(jmp_buf); }

/*
 * `mem.par_for`: runs `body(env, lo, hi)` over `[0, n)` split into one chunk per thread.
 * Threads are created on first use; T is MIM_NUM_THREADS, else the core count. The calling thread
 * runs chunk 0 itself. No Windows threads yet: there every chunk runs on the caller.
 */

typedef void (*mim_rt_par_body)(void* env, int64_t lo, int64_t hi);

#if defined(_WIN32)

void mim_rt_parallel_for(int64_t n, mim_rt_par_body body, void* env) { body(env, 0, n); }

#else

#    include <pthread.h>
#    include <stdlib.h>
#    include <unistd.h>
#    if defined(__APPLE__)
#        include <sys/sysctl.h>
#    endif

static struct {
    pthread_t* threads;
    int T;
    pthread_mutex_t mu;
    pthread_cond_t start, done;
    unsigned gen;
    int pending;
    int64_t tasks, chunk, n;
    mim_rt_par_body body;
    void* env;
} pool = {.mu = PTHREAD_MUTEX_INITIALIZER, .start = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER};

static int num_threads(void) {
    const char* s = getenv("MIM_NUM_THREADS");
    if (s && atoi(s) > 0) return atoi(s);
    long n = 0;
#    if defined(__APPLE__)
    size_t len = sizeof n; // the performance cores; the efficiency cores would hold a static split back
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, NULL, 0) != 0) n = 0;
#    endif
    if (n <= 0) n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

static void run_task(int64_t id) {
    int64_t lo = id * pool.chunk, hi = lo + pool.chunk;
    pool.body(pool.env, lo, hi < pool.n ? hi : pool.n);
}

static void* worker(void* arg) {
    int64_t id   = (int64_t)(intptr_t)arg;
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&pool.mu);
        while (pool.gen == seen) pthread_cond_wait(&pool.start, &pool.mu);
        seen = pool.gen;
        pthread_mutex_unlock(&pool.mu);
        if (id < pool.tasks) {
            run_task(id);
            pthread_mutex_lock(&pool.mu);
            if (--pool.pending == 0) pthread_cond_signal(&pool.done);
            pthread_mutex_unlock(&pool.mu);
        }
    }
    return NULL;
}

static void init_pool(void) {
    pool.T       = num_threads();
    pool.threads = (pthread_t*)calloc(pool.T > 1 ? pool.T - 1 : 1, sizeof(pthread_t));
    for (int i = 1; i < pool.T; ++i)
        if (pthread_create(&pool.threads[i - 1], NULL, worker, (void*)(intptr_t)i) != 0) {
            pool.T = i;
            break;
        }
}

void mim_rt_parallel_for(int64_t n, mim_rt_par_body body, void* env) {
    if (pool.T == 0) init_pool();
    if (n <= 1 || pool.T <= 1) {
        if (n > 0) body(env, 0, n);
        return;
    }
    pthread_mutex_lock(&pool.mu);
    pool.chunk   = (n + pool.T - 1) / pool.T;
    pool.tasks   = (n + pool.chunk - 1) / pool.chunk;
    pool.n       = n;
    pool.body    = body;
    pool.env     = env;
    pool.pending = (int)pool.tasks - 1;
    ++pool.gen;
    pthread_cond_broadcast(&pool.start);
    pthread_mutex_unlock(&pool.mu);
    run_task(0);
    pthread_mutex_lock(&pool.mu);
    while (pool.pending != 0) pthread_cond_wait(&pool.done, &pool.mu);
    pthread_mutex_unlock(&pool.mu);
}

#endif
