// Runtime wrappers for the MimIR `ll` backend.
//
// The functions in this file are implemented in C, compiled to textual LLVM IR by `clang`
// at build time (see `add_mim_runtime` in `cmake/Mim.cmake`), and then either embedded into
// or linked with the module emitted by the `ll` backend (selected via `-X ll:rt=embed|extern`).
//
// This lets the emitter offload complex or platform-dependent lowerings to `clang` instead of
// hand-writing raw LLVM IR; see issue #486.
// Wrappers must expose a flat, scalar ABI (no C aggregates across the boundary) so that the
// emitter can lower a Mim intrinsic to a single `call`.

#include <setjmp.h>
#include <stdint.h>
#include <stdlib.h>

/// Size in bytes of a `jmp_buf`, used by `clos.alloc_jmpbuf` to reserve stack space.
/// The size is platform- and libc-dependent, so we let the C compiler compute it rather than
/// hard-coding it in the backend.
int64_t mim_jmpbuf_size(void) { return (int64_t)sizeof(jmp_buf); }

/// Backs a `mem.persist` call site: `slot` holds the source of the previous call and the storage, which it returns.
/// The storage is allocated anew, and `*fresh` set, on the first call and whenever `src` differs.
/// The caller reads the storage only through the result, so that, as for `malloc`, it aliases nothing the caller holds.
__attribute__((noinline, malloc)) void* mim_persist(void** slot, void* src, int64_t size, int64_t* fresh) {
    *fresh = !slot[1] || slot[0] != src;
    if (*fresh) {
        free(slot[1]);
        slot[0] = src;
        slot[1] = aligned_alloc(64, (size + 63) / 64 * 64);
    }
    return slot[1];
}
