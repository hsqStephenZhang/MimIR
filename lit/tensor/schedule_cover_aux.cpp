// Driver for schedule_cover.mim: checks each result against a scalar reference.
// Exit code: 0 if all match, 1 otherwise.

#include <cstdio>
#include <cstdlib>

#include <utility>

struct Pair {
    int* d;
    int* e;
};
extern "C" {
Pair diag(int* q);
int* rows(int* a);
int* rows_i(int* a);
int* rows_t(int* a);
}

static int report(const char* name, int bad) {
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    int bad = 0, diag_bad = 0;
    int q[4 * 4 * 3], a[6 * 10];
    for (int i = 0; i < 48; ++i)
        q[i] = (7 * i) % 11 - 5;
    for (int i = 0; i < 60; ++i)
        a[i] = (3 * i) % 13 - 6;
    auto [d, e] = diag(q);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int p = 0;
            for (int k = 0; k < 3; ++k)
                p += q[(i * 4 + j) * 3 + k];
            diag_bad += (i == j && d[i] != p + 1) + (e[i * 4 + j] != p + 1);
        }
    bad += report("diag", diag_bad);
    for (auto [name, f] : {
             std::pair{  "rows",   rows},
             std::pair{"rows_i", rows_i},
             std::pair{"rows_t", rows_t}
    }) {
        int* r       = f(a);
        int rows_bad = 0;
        for (int i = 0; i < 6; ++i) {
            int s = 0;
            for (int k = 0; k < 10; ++k)
                s += a[i * 10 + k];
            rows_bad += r[i] != s;
        }
        bad += report(name, rows_bad);
    }
    return bad != 0;
}
