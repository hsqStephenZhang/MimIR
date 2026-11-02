#include <cstdint>

extern "C" bool dead_query(uint64_t);

int main() { return dead_query(3) || dead_query(0); }
