#include <cassert>
#include <cstdint>
#include <cstdlib>

struct RowsPair {
    int64_t* rows;
    int64_t* elements;
};
struct ElementsPair {
    int64_t* elements;
    int64_t* rows;
};

extern "C" int64_t* rowsum(int64_t*, int64_t*);
extern "C" RowsPair two(int64_t*, int64_t*);
extern "C" ElementsPair swapped(int64_t*, int64_t*);
extern "C" ElementsPair whole_reader(int64_t*, int64_t*);
extern "C" int64_t* swiglu(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* permuted(int64_t*, int64_t*);
extern "C" int64_t* epilogue(int64_t*, int64_t*);

static uint64_t multiplies;
extern "C" void record_multiply(uint64_t lanes) { multiplies += lanes; }

void check_work(uint64_t expected) {
#ifdef REVERSE_COUNT
    assert(multiplies == expected);
#endif
    multiplies = 0;
}

int main() {
    int64_t a[15], b[21], c[21], p[35], u[35], sums[5]{};
    for (int attempt = 0; attempt != 2; ++attempt) {
        for (int i = 0; i != 15; ++i)
            a[i] = (i * 3 + attempt) % 13 - 6;
        for (int i = 0; i != 21; ++i) {
            b[i] = (i * 2 + attempt) % 9 - 4;
            c[i] = (i * 5 + attempt) % 11 - 5;
        }
        for (int i = 0; i != 5; ++i) {
            sums[i] = 0;
            for (int j = 0; j != 7; ++j) {
                p[i * 7 + j] = u[i * 7 + j] = 0;
                for (int k = 0; k != 3; ++k) {
                    p[i * 7 + j] += a[i * 3 + k] * b[k * 7 + j];
                    u[i * 7 + j] += a[i * 3 + k] * c[k * 7 + j];
                }
                sums[i] += p[i * 7 + j];
            }
        }
        auto s = rowsum(a, b);
        check_work(105);
        auto first = two(a, b);
        check_work(105);
        auto second = swapped(a, b);
        check_work(105);
        auto whole = whole_reader(a, b);
        check_work(105);
        auto h = swiglu(a, b, c);
        check_work(245);
        auto transposed = permuted(a, b);
        check_work(105);
        auto doubled = epilogue(a, b);
        check_work(105);
        for (int i = 0; i != 5; ++i) {
            assert(s[i] == sums[i]);
            assert(first.rows[i] == sums[i]);
            assert(second.rows[i] == sums[i]);
            assert(whole.rows[i] == sums[i]);
        }
        for (int i = 0; i != 35; ++i) {
            assert(whole.elements[i] == p[i]);
            assert(first.elements[i] == p[i] + 1);
            assert(second.elements[i] == p[i] + 1);
            assert(transposed[(i % 7) * 5 + i / 7] == p[i] + 1);
            assert(doubled[i] == 2 * p[i] + 1);
            assert(h[i] == p[i] * u[i]);
        }
        std::free(s);
        std::free(first.rows);
        std::free(first.elements);
        std::free(second.rows);
        std::free(second.elements);
        std::free(whole.rows);
        std::free(whole.elements);
        std::free(h);
        std::free(transposed);
        std::free(doubled);
    }
}
