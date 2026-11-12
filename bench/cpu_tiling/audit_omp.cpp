// Untimed audit only. Compile mim_omp_rt.c with -Dmim_par_for=mim_par_for_impl.
#include <cassert>
#include <cstdint>

#include <atomic>

#include <sched.h>
extern "C" int omp_get_thread_num();
extern "C" int omp_get_num_threads();
using Body = void (*)(void*, int64_t, int64_t);
extern "C" void mim_par_for_impl(int64_t, int64_t, int64_t, Body, void*);
static std::atomic<uint64_t> workers{0}, cpus{0};
static std::atomic<int> team{0}, launches{0};
struct Context {
    Body body;
    void* env;
};
static void chunk(void* raw, int64_t lo, int64_t hi) {
    auto* context = static_cast<Context*>(raw);
    workers.fetch_or(uint64_t(1) << omp_get_thread_num());
    int cpu = sched_getcpu();
    assert(cpu >= 0 && cpu < 64);
    cpus.fetch_or(uint64_t(1) << cpu);
    team.store(omp_get_num_threads());
    context->body(context->env, lo, hi);
}
extern "C" void mim_par_for(int64_t lo, int64_t hi, int64_t step, Body body, void* env) {
    ++launches;
    Context context{body, env};
    mim_par_for_impl(lo, hi, step, chunk, &context);
}
extern "C" uint64_t audit_workers() { return workers.load(); }
extern "C" uint64_t audit_cpus() { return cpus.load(); }
extern "C" int audit_team() { return team.load(); }
extern "C" int audit_launches() { return launches.load(); }
