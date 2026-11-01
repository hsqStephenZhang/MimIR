#include <assert.h>
#include <stdint.h>
#include <string.h>

extern void acc(void*, uint32_t);

int main(void) {
    _Alignas(64) unsigned char storage[16 * sizeof(float) + 1];
    void* p = storage + 1; // Byte alignment exercises the emitted align 1 without unaligned C float accesses.
    const uint32_t counts[] = {0, 1, 15, 16, 17, 32, 33};
    for (uint32_t c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c) {
        uint32_t n = counts[c];
        for (uint32_t i = 0; i < 16; ++i) {
            float value = (float)(i + 1);
            memcpy(storage + 1 + i * sizeof(float), &value, sizeof(value));
        }
        acc(p, n);
        for (uint32_t i = 0; i < 16; ++i) {
            float value;
            memcpy(&value, storage + 1 + i * sizeof(float), sizeof(value));
            assert(value == (float)((i + 1) * (1 + n / 16 + (i < n % 16))));
        }
    }
    return 0;
}
