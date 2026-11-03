#include <cassert>
#include <cstdint>
#include <cstdlib>

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

static int packs;
static size_t packed_writes;
static std::vector<std::pair<uintptr_t, size_t>> packed_buffers;

extern "C" void* record_pack(size_t alignment, size_t size) {
    ++packs;
    void* pointer;
    assert(posix_memalign(&pointer, alignment, size) == 0);
    packed_buffers.emplace_back(reinterpret_cast<uintptr_t>(pointer), size);
    return pointer;
}

extern "C" void record_free(void* pointer) {
    std::erase_if(packed_buffers, [&](auto buffer) { return buffer.first == reinterpret_cast<uintptr_t>(pointer); });
    std::free(pointer);
}

extern "C" void record_packed_write(void* pointer) {
    auto address = reinterpret_cast<uintptr_t>(pointer);
    for (auto [base, size] : packed_buffers)
        if (address >= base && address - base < size) {
            ++packed_writes;
            return;
        }
}
