#include <cassert>
#include <cstdlib>

extern "C" float packed(float*);
static int packs;
extern "C" void* record_pack(size_t alignment, size_t size) {
    ++packs;
    void* pointer;
    assert(posix_memalign(&pointer, alignment, size) == 0);
    return pointer;
}

int main() {
    float a[35], b[35];
    for (int i = 0; i != 35; ++i) {
        a[i] = i;
        b[i] = -i;
    }
    assert(packed(a) == 34);
    assert(packs == 1);
    assert(packed(a) == 34);
    assert(packs == 1);
    assert(packed(b) == -34);
    assert(packs == 2);
    assert(packed(a) == 34);
    assert(packs == 3);
}
