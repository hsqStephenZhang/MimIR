#include <cassert>
#include <cstdint>

extern "C" int64_t fresh_alloc();

int main() { assert(fresh_alloc() == 42); }
