# Non-libdevice FP32/F64 math

## Examples-only branch

This directory supplies optional benchmark tooling, not MimIR compiler changes.
The recorded GPU results used a locally modified compiler; those fixes are not
included in this examples-only branch.
Full execution requires a compatible NVPTX backend that honors `Xlink_llvm` and
`Xopt` even without libdevice, plus the LLVM representation fixes used in the
original experiment.
The branch's unmodified compiler base is not claimed to pass all 72 cases.
Use `run.py --mim <compatible-mim>` for execution, or `run.py --emit-only` to
generate the Mim sources without a compiler or CUDA tools.
References below to emitter behavior describe the tested compiler configuration.

The Mim NVPTX emitter no longer intercepts `math::pow` or `math::er`.
`pow` remains `llvm.pow.*` in emitted LLVM IR, and `erf` remains a normal math-library call.
This optional benchmark toolchain supplies FP32 `powf`/`erff` and F64 `pow`/`erf`/`erfc` without NVIDIA libdevice.
It does not replace existing exp/trigonometric handling elsewhere in the backend.

## Pipeline

```text
Mim pow -> llvm.pow.f32/f64 -> default<O2>
    -> LLVM IntrinsicLowering -> ordinary powf/pow call
    -> LLVM libc generic implementation -> NVPTX

Mim erf -> ordinary erff call -> LLVM libc generic implementation -> NVPTX
Mim F64 erf/erfc -> ordinary erf/erfc call -> OpenLibm implementation -> NVPTX
```

`lower_math.cpp` delegates scalar FP32/F64 pow legalization to LLVM's `IntrinsicLowering`.
It does not implement a power approximation or override Mim's math semantics.
The pass runs after ordinary LLVM optimization so that the intrinsic is available to optimizers first.
The software implementations use LLVM libc and OpenLibm sources pinned in `sources.json`.
LLVM libc 22.1.3 lacks a generic double erf implementation; OpenLibm supplies double erf/erfc, with LLVM libc providing their exp/fabs dependencies.
No `__nv_pow*` or `__nv_erf*` mapping is used.

The FP16 operation variants explicitly compute math in FP32.
FP32 erfc is not supplied by this runtime.
The bitcode is compiled without fast-math and without CUDA's GPU library (`-nogpulib`).
OpenLibm is compiled with floating-point contraction disabled and its host weak-alias macro disabled because NVPTX does not accept ELF `.weak`/`.equ` assembly.

## Build and run

Run from the isolated MimIR worktree root with matching LLVM/Clang 22 development tools, CUDA tools and a working `build-clang`:

```bash
python3 bench/leetcuda-simt/toolchain/build.py
python3 bench/leetcuda-simt/run.py
python3 bench/leetcuda-simt/run.py --filter '^(math_pow|math_erf|gelu_erf|rope)$' --sanitize memcheck
python3 bench/leetcuda-simt/run.py --filter '^math_.*_f64$' --sanitize memcheck
```

The build downloads declared dependencies into ignored `artifacts/llvm-project` and `artifacts/openlibm`.
It verifies their pinned commits and clean tracked sources, then builds an LLVM pass plugin and a software-math bitcode module under `artifacts/math-toolchain`.
It neither patches nor replaces system LLVM binaries.
LLVM libc retains its Apache-2.0 WITH LLVM-exception license in that dependency checkout; its implementation is not copied into MimIR.
OpenLibm's upstream notices, including the Sun permissive notice in `s_erf.c`, remain in its dependency checkout.

## F64 pow corrections

GPU tests exposed two numerical bugs in LLVM libc 22.1.3's double pow.
`pow-subnormal.patch` is applied only to the generated build copy of `pow.cpp`; the pinned checkout stays unmodified:

- Subnormal-input normalization extracted the scaled mantissa but retained the original exponent minus 64. It must instead use the scaled exponent minus 64. The regression includes `pow(0x0.2100000000000p-1022, -0.875)`, whose unpatched result was approximately `2.02051e286` rather than `9.43157e269`.
- The `y = -2` shortcut evaluated `1/(x*x)`, losing representable subnormal outputs when the intermediate square overflowed. Negative two now uses the existing general pow algorithm; `pow(0x1p512, -2)` must produce `0x0.4p-1022`, not zero.

The patch and toolchain binaries are hashed in the build and result manifests.
These are accuracy fixes to the standard implementation, not an exp/log substitution in Mim kernels.

The build rejects unresolved software-runtime symbols or `__nv_*` references.
It also verifies the standalone intrinsic-to-call transformation and compiles `pow.ll` to a cubin without linking libdevice.
`manifest.json` records the toolchain inputs and binary hashes.
The benchmark adds the device library through `-Xll_nvptx:Xlink_llvm=...` and selects the late pass through `-Xll_nvptx:Xopt=...`.
Missing runtime files are an explicit error; there is no vendor fallback.

## LLVM-only reproduction

`pow.ll` is a minimal, valid LLVM IR kernel independent of MimIR:

```bash
opt -passes=verify bench/leetcuda-simt/toolchain/pow.ll -o /tmp/mim-pow.bc
llc -march=nvptx64 -mcpu=sm_75 /tmp/mim-pow.bc -o /tmp/mim-pow.ptx
```

Unmodified LLVM 22.1.3 fails with `Cannot select ... fpow`.
The observed diagnostic is saved in `artifacts/math-toolchain/raw-llc.log`.
The same input compiles after standard intrinsic lowering and linking the LLVM libc implementation; the resulting `probe.ptx` contains no `__nv_*` references.

## Validation

The 72-case GPU suite includes dedicated FP32 pow/erf kernels with two-ULP checks against a double CPU reference rounded to FP32.
Three additional native F64 pow/erf/erfc kernels each check 8,193 outputs across three fixtures.
F64 input/output buffers are actual doubles, and their two-ULP oracle uses host long-double math rounded to double.
The harness rejects hosts whose long double is not wider than double; this host has a 64-bit long-double significand versus double's 53 bits.
The JSON records the maximum observed ULP error and reference precision.
Inputs include negative bases, fractional/integral exponents, NaNs, infinities, signed zeros, subnormals, overflow and underflow.
F64 cases additionally span the binary exponent range and erf/erfc approximation boundaries with neighboring representable inputs.
These are focused checks, not an exhaustive proof of all FP32/F64 inputs or an MPFR oracle.
The affected GELU-erf and RoPE FP32/FP16-storage variants retain their original formulas and pass the full-output operation checks.
All 72 cases pass memcheck; the 12 cooperative cases also pass racecheck and synccheck.

RoPE still uses the pre-existing libdevice sin/cos path.
Its PTX can therefore contain unused exported vendor math definitions from that library; the relevant check is that it does not call vendor pow/erf, not that every vendor symbol disappears.
