CPU thread tiling
=================

`-p omp` loads the fork/join outlining and runtime support from `origin/feature/omp`
(277874c4756a). Automatic tensor GEMM scheduling adds an outer `TileLevels#Threads`
level around the existing serial NC/KC/MC packing, reduction and register levels.
It does not parallelize each scalar or microkernel loop.

Use `-X omp:threads=4` when lowering and `OMP_NUM_THREADS=4` when executing.
The compile-time hint chooses a balanced two-dimensional M/N task grid. N tile
boundaries respect NR, and packing volume breaks ties. The runtime partitions the
linearized grid into contiguous chunks; the team size may differ from the hint
without changing the result. Keep them equal for performance measurements.

Only static 2D contractions receive automatic thread tiling. The default threshold
is 1,048,576 multiply-adds (`-X omp:threshold=N`); smaller contractions remain
serial. `-X omp:threads=1` disables automatic thread tiling. Explicit schedules use
`tensor.thread_par (sizes, 0, inner_levels)`; reduction tile sizes must remain zero,
output extents must be literal, and partial output tiles use capped tails (0).
Nested thread levels and mixing thread/GPU bindings are rejected. Without `omp`,
explicit thread levels lower to serial loops.

The output is allocated before the parallel region. Each tile owns its temporary
panels and fold buffers; scratch allocation is never hoisted across the thread
boundary. The memory token is private to the outlined chunk; only pointers and
scalar values are captured. `-X omp:strict=1` rejects outlining failures instead
of warning and executing serially.

The default LLVM output embeds `omp_rt.ll`; link it with `clang++ -fopenmp`.
Alternatively use `-X ll:rt=extern`, compile `src/mim/plug/omp/rt/mim_omp_rt.c`
with GCC/Clang OpenMP, and link the resulting object with the matching runtime.
Use one runtime per process. In particular PyTorch wheels can carry a private
libgomp; the remote benchmark explicitly links that same library.

Reproduction scripts:

* `prepare_omp.py`: lower gemm-bench nominal-tensor cases with an explicit hint and
  Haswell 6x16/KC=320/MC=320 parameters; save source identity, commands and IR hashes.
* `remote_cpu.py`: compile locally lowered IR on the target, validate all kernels
  against FP64, then time gemm-bench's MimIR, PyTorch and OpenBLAS native wrappers.
  `--threads T --cpus comma,separated,physical,cpus` sets affinity before importing
  numerical libraries. For `--runtime-source`, also supply `--omp-runtime`.

Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` consistently.
Use one socket, distinct physical cores, `MKL_DYNAMIC=FALSE`, `OMP_DYNAMIC=FALSE`,
`OMP_PROC_BIND=FALSE` (so later pthread pools inherit the full mask),
`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, and `OPENBLAS_THREAD_TIMEOUT=4`.
The last three prevent idle pools from competing with the timed backend. The
OpenBLAS timeout is a log2 cycle count clamped to at least 4, not a duration.
AVX2 experiments additionally use `--isa avx2` and the dispatch controls recorded
in the result metadata. All compilation/validation finishes before measurement;
9 shuffled rounds use native batches of at least 100ms and report medians.
