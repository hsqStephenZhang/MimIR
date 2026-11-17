B = 'bench/typed_tiles/'
pos = open(B + 'gemm_block_steps_threads.mim').read()
open('lit/gpu/typed_tiles.mim', 'w').write('''// RUN: %mim -p opt -X tensor:tile-fuse=1 %s -p gpu -o - | %FileCheck %s

// Tiles typed by their owner, lifetime, memory space, what may be done with them and how they are stored, over the
// tile primitives: a work-group's blocks
// stage per-step panels its work-items read, one of them column-major, and the indices leave nothing behind once the
// schedule is unfolded.
// `lit/error/typed_tile_*.mim` are schedules these types reject.

''' + pos.rstrip() + '''

// CHECK-NOT: nom Tile
// CHECK-NOT: nom Ctx
// CHECK-NOT: gpu.Vis
// CHECK: gpu.launch
''')
open('lit/gpu/typed_tiles_aux.cpp', 'w').write(open(B + 'driver_gemm.cpp').read())
open('lit/nvptx/typed_tiles.mim', 'w').write('''// RUN: rm -f %{s:stem}.ll %{s:stem}_dev.ll %t
// RUN: %mim -p opt -X tensor:tile-fuse=1 %S/../gpu/%{s:stem}.mim -p nvptx
// RUN: %FileCheck %s --input-file=%{s:stem}_dev.ll
// RUN: %if system-linux && cuda %{ clang++ -O1 %S/../gpu/%{s:stem}_aux.cpp %{s:stem}.ll -lcuda -o %t -Wno-override-module %}
// RUN: %if system-linux && cuda %{ %t %}

// The typed schedule of `lit/gpu/typed_tiles.mim` runs as one kernel whose work-groups stage panels behind barriers.

// CHECK: define ptx_kernel void @tileKernel
// CHECK: @llvm.nvvm.barrier0
''')
panel = open(B + 'gemm_panel_placed.mim').read()
open('lit/gpu/typed_panel.mim', 'w').write('''// RUN: %mim -p opt -X tensor:tile-fuse=1 %s -p gpu -o - | %FileCheck %s

// The schedule of `lit/gpu/typed_tiles.mim` with A's panel staged padded: its readers use logical coordinates.

''' + panel.rstrip() + '''

// CHECK-NOT: nom Tile
// CHECK: gpu.launch
''')
open('lit/nvptx/typed_panel.mim', 'w').write('''// RUN: rm -f %{s:stem}.ll %{s:stem}_dev.ll %t
// RUN: %mim -p opt -X tensor:tile-fuse=1 %S/../gpu/%{s:stem}.mim -p nvptx
// RUN: %FileCheck %s --input-file=%{s:stem}_dev.ll
// RUN: %if system-linux && cuda %{ clang++ -O1 %S/../gpu/typed_tiles_aux.cpp %{s:stem}.ll -lcuda -o %t -Wno-override-module %}
// RUN: %if system-linux && cuda %{ %t %}

// A's panel of `lit/gpu/typed_panel.mim` lies in shared memory as its layout says, 8 rows padded to 12 elements.

// CHECK: define ptx_kernel void @tileKernel
// CHECK: [8 x [12 x float]]
''')
what = {
    'n6': ('step', "A panel staged for one reduction step is read in the context of another."),
    'n7': ('unit', "A work-group claims a tile in a work-item's registers."),
    'n8': ('lifetime', "A box staged for the work-group's whole life is handed out as a step's panel."),
    'n9': ('seed', "An unordered split takes the seed, not an accumulator a fold has touched."),
    'n11': ('owner', "A work-item wraps its partial as another work-item's."),
    'n12': ('exclusive', "A work-item reads the accumulator its work-group alone may touch."),
    'n13': ('readonly', "A work-item takes a published panel as a tile of its own to write."),
    'n14': ('layout', "A panel stored column-major is handed over as a row-major one."),
    'n15': ('vectors', "A vector view whose vectors do not divide the stored last axis."),
    'n16': ('panel_layout', "A panel staged in one layout is handed over as a panel in another."),
}
for n, (name, doc) in what.items():
    s = open(B + f'reject_{n}.mim').read()
    open(f'lit/error/typed_tile_{name}.mim', 'w').write(
        '// RUN: not %mim -p opt -X tensor:tile-fuse=1 %s -p gpu -o /dev/null 2>&1 | %FileCheck %s\n'
        + ('// CHECK: error: a vector view needs whole vectors\n\n' if n == 'n15' else '// CHECK: error: {{.*}}not assignable\n\n') +
        f'// {doc}\n// The combinators are those of `lit/gpu/typed_tiles.mim`.\n\n' + s)
