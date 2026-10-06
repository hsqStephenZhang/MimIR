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

void mim_rt_parallel_for(int64_t n, mim_rt_par_body body, void* env) {
    if (n > 0) body(env, 0, n);
}

#else

#    include <limits.h>
#    include <pthread.h>
#    include <stdlib.h>
#    include <unistd.h>
#    if defined(__APPLE__)
#        include <sys/sysctl.h>
#    endif

// A synchronous call owns this descriptor until its last worker completes. Independent callers
// serialize their submissions; nested calls run inline instead of waiting on their own pool.
struct par_job {
    mim_rt_par_body body;
    void* env;
    int64_t n;
    int tasks, pending;
};

static struct {
    int T;
    pthread_mutex_t submit, mu;
    pthread_cond_t start, done;
    uint64_t gen;
    struct par_job* job;
} pool = {.submit = PTHREAD_MUTEX_INITIALIZER,
          .mu     = PTHREAD_MUTEX_INITIALIZER,
          .start  = PTHREAD_COND_INITIALIZER,
          .done   = PTHREAD_COND_INITIALIZER};

static pthread_once_t pool_once = PTHREAD_ONCE_INIT;
static _Thread_local unsigned parallel_depth;

static int num_threads(void) {
    const char* s = getenv("MIM_NUM_THREADS");
    if (s) {
        char* end;
        long t = strtol(s, &end, 10);
        if (s != end && *end == '\0' && 0 < t && t < INT_MAX) return (int)t;
    }
    long n = 0;
#    if defined(__APPLE__)
    int cores  = 0; // the performance cores; the efficiency cores would hold a static split back
    size_t len = sizeof cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &len, NULL, 0) == 0) n = cores;
#    endif
    if (n <= 0) n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

static void run_body(mim_rt_par_body body, void* env, int64_t lo, int64_t hi) {
    ++parallel_depth;
    body(env, lo, hi);
    --parallel_depth;
}

static void run_task(const struct par_job* job, int id) {
    // Balanced nonempty chunks, without overflowing even when n == INT64_MAX.
    int64_t q = job->n / job->tasks, r = job->n % job->tasks;
    int64_t lo = id * q + (id < r ? id : r);
    run_body(job->body, job->env, lo, lo + q + (id < r));
}

static void* worker(void* arg) {
    int id        = (int)(intptr_t)arg;
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&pool.mu);
        while (pool.gen == seen)
            pthread_cond_wait(&pool.start, &pool.mu);
        seen                = pool.gen;
        struct par_job* job = pool.job;
        // Check participation under the lock. An inactive worker may skip generations, but must
        // never read a newer job after remembering an older generation.
        if (!job || id >= job->tasks) {
            pthread_mutex_unlock(&pool.mu);
            continue;
        }
        pthread_mutex_unlock(&pool.mu);
        run_task(job, id);
        pthread_mutex_lock(&pool.mu);
        if (--job->pending == 0) pthread_cond_signal(&pool.done);
        pthread_mutex_unlock(&pool.mu);
    }
    return NULL;
}

static void init_pool(void) {
    int requested = num_threads();
    pool.T        = 1;
    for (int i = 1; i < requested; ++i) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, worker, (void*)(intptr_t)i) != 0) break;
        pthread_detach(thread);
        ++pool.T;
    }
}

void mim_rt_parallel_for(int64_t n, mim_rt_par_body body, void* env) {
    if (n <= 0) return;
    if (parallel_depth || n == 1) {
        run_body(body, env, 0, n);
        return;
    }
    pthread_once(&pool_once, init_pool);
    if (pool.T <= 1) {
        run_body(body, env, 0, n);
        return;
    }
    pthread_mutex_lock(&pool.submit);
    int tasks          = n < pool.T ? (int)n : pool.T;
    struct par_job job = {body, env, n, tasks, tasks - 1};
    pthread_mutex_lock(&pool.mu);
    pool.job = &job;
    ++pool.gen;
    pthread_cond_broadcast(&pool.start);
    pthread_mutex_unlock(&pool.mu);
    run_task(&job, 0);
    pthread_mutex_lock(&pool.mu);
    while (job.pending != 0)
        pthread_cond_wait(&pool.done, &pool.mu);
    pool.job = NULL;
    pthread_mutex_unlock(&pool.mu);
    pthread_mutex_unlock(&pool.submit);
}

#endif
