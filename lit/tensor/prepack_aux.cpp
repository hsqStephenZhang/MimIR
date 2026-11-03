#include "prepack_counter.h"

extern "C" float packed(float*);

int main() {
    float a[35], b[35];
    for (int i = 0; i != 35; ++i) {
        a[i] = i;
        b[i] = -i;
    }
    assert(packed(a) == 34);
    assert(packs == 1);
    assert(packed_writes == 40);
    assert(packed(a) == 34);
    assert(packs == 1);
    assert(packed_writes == 40);
    assert(packed(b) == -34);
    assert(packs == 2);
    assert(packed_writes == 80);
    assert(packed(a) == 34);
    assert(packs == 3);
    assert(packed_writes == 120);
}
