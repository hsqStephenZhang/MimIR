# Research question: is a linear `RingState` feasible in principle for MimIR's typed GPU ring protocols?

This is ordinary research on our own compiler. Read-only: do not modify, commit or push anything. Write your answer in Chinese.

## Positioning
MimIR is a compiler, not a kernel DSL. Users write PyTorch models and the compiler generates the GPU kernels. The protocol typing checks the compiler's own generated code (certifying compilation / translation validation). Writing kernels by hand in Mim is a secondary use.

## What RingState is meant to do
- Ring slot tokens `Held (rid, depth) (s, live)` and `Vacant (rid, depth) s` are checked after partial evaluation by `gpu.check_sync`.
- Still accepted: awaiting the same slot twice yields two live tokens for one slot.
- The plan (design doc §14.1 item 3, §14.2 "替代方案", §17) is a linear protocol state `RingState rid` threaded through every operation of `SyncOf` (open, acquire, commit, await, release, settle). It carries the producer and consumer cursors, the slot phases and the groups in flight, so that a token can only come from a transition that consumes the unique prior state. The user chose "capability + residual validation".
- **The blocker (§17, §21.3):**
  - The ring loops are `affine.For` loops whose state type cannot depend on the iteration number.
  - A state indexed by its cursor, e.g. `RingState rid n`, or a slot handle rotated by `n` (the rolled ring and `ws_fold` rotate the slot array each step and read position 0, which is physical slot `block % depth`), therefore cannot be the loop's carried type.
  - Switching to dynamic slot indexing would change the generated device IR, which we want to keep byte-identical.

## Read first
- Worktree `/home/chzh00001/workspaces/MimIR-types`, branch `feat/protocol-types`.
- `docs/linear-types-design.zh-CN.md`, in Chinese: §4–§7 design, §14 the earlier review, §17 decisions, §20–§21 status.
- `src/mim/plug/gpu/gpu.mim`, around lines 860–1190: `Held`, `Vacant`, `Panels`, `read_slot`, `SyncOf`, `ring_fold_at` (rolled and turns variants), `ws_fold_at`, and the ring syncs.
- `src/mim/plug/gpu/phase/check_sync.cpp`: the post-PE checker.
- `src/mim/plug/affine/affine.mim` (`affine.For`) and the core: `include/mim/def.h`, `include/mim/lam.h`, `src/mim/check.cpp`. These show what dependent types MimIR supports (dependent Pi/Sigma, `Arr`, mutable binders, holes).
- `proofs/ring/README.md`: the Lean model and its theorems.
- Papers, local copies:
  - `/home/chzh00001/async_graphene.pdf`
  - `/home/chzh00001/workspaces/MimIR/cgo27-paper248.pdf`, the base MimIR multi-level paper.
  - Also search the web.

## Questions
1. **In principle.** Can a linear, sealed `RingState` that rules out double `await` and pins tokens to the right slot and generation be expressed in MimIR's type theory?
   - Options to consider:
     - an existential or dependent pair in the loop state, e.g. `Σ n. RingState rid n` with a runtime-free ghost;
     - singleton or refinement types relating the loop counter to the state index;
     - an indexed loop combinator replacing `affine.For` for rings, i.e. a dependently typed fold or recursor;
     - typestate as a linear capability with erased indices checked post-PE only;
     - leaving the cursor arithmetic to residual validation, i.e. the Lean certificate checker on extracted traces, and typing only the per-step local discipline.
   - For each, say what MimIR core would need, whether device IR stays identical after erasure, and what is checked statically vs by validation.
2. **Prior work.**
   - Survey how others type such protocols: linear/affine typestate for asynchronous resources and barriers; indexed and session types for ring buffers or channels with bounded capacity; dependent folds and recursors carrying iteration-indexed state.
   - Candidates (verify, don't trust this list):
     - Async Graphene (CC'26); Descend (PLDI'24); Tawa / aref; ThunderKittens; CuTe/CUTLASS pipelines.
     - Rust GPU or embassy-style DMA typestate; ATS; Mezzo; Alms; Linear Haskell; Granule / QTT (Idris 2).
     - Low*/F* buffer and modifies clauses; RustBelt and Iris ghost state; VerCors / GPUVerify barrier invariants; Kuiper.
     - Koska et al., arXiv:2611.04447 (Lean, Hopper GEMM); session types with bounded buffers; "Linear types can change the world"; Liquid Haskell or refinement loop invariants.
   - Which projects actually implement something like an indexed linear ring state, and how do they handle loops whose state index advances each iteration?
3. **Recommendation.**
   - The most defensible design for a paper whose contribution is typing compiler-generated GPU protocols, under the constraint of unchanged device IR.
   - A concrete minimal next step in MimIR, with the files to change.
   - What the type system then guarantees versus what the Lean residual validation must cover.

Cite file:line for code and full references for papers. Mark speculation as such. Rank the options.
