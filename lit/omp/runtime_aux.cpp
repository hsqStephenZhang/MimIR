#include <cassert>
#include <cstdint>

#include <atomic>

#include <omp.h>
extern "C" void mim_par_for(int64_t, int64_t, int64_t, void (*)(void*, int64_t, int64_t), void*);
struct State {
    std::atomic<unsigned> mask{0};
    std::atomic<int> visits[97]{};
};
static void body(void* raw, int64_t lo, int64_t hi) {
    auto& s = *static_cast<State*>(raw);
    s.mask.fetch_or(1u << omp_get_thread_num());
    assert(hi <= 97);
    for (auto i = lo; i < hi; i += 3)
        ++s.visits[i];
}
static void nested(void* raw, int64_t lo, int64_t hi) {
    assert(omp_in_parallel());
    mim_par_for(lo, hi, 3, body, raw);
}
int main() {
    State s;
    mim_par_for(1, 97, 3, body, &s);
    assert(s.mask.load() == ((1u << omp_get_max_threads()) - 1));
    for (int i = 0; i < 97; ++i)
        assert(s.visits[i] == (i >= 1 && (i - 1) % 3 == 0));
    State empty;
    mim_par_for(4, 4, 1, body, &empty);
    mim_par_for(0, 4, 0, body, &empty);
    assert(empty.mask == 0);
    State inner;
    mim_par_for(1, 97, 3, nested, &inner);
    for (int i = 0; i < 97; ++i)
        assert(inner.visits[i] == (i >= 1 && (i - 1) % 3 == 0));
}
