# A Lean model of `SchedTree` and `run_levels_at`

A model of the abstract machine of `src/mim/plug/tensor/tensor.mim` with the proofs that `docs/pe-cpu-tiling-formalization.zh-CN.md` plans.
It needs Lean 4.36 and nothing else; build it with `lake build`.

## What is proved

| File | Theorem | Statement |
| :--- | :--- | :--- |
| `Sched/Aff.lean` | `affBox_sound` | Every read of a tile falls into the box `aff_box` derives, for all eight constructors of `AffE`, and an affine row reads the box at `aff_rel`. |
| `Sched/Aff.lean` | `AffE.span_le_reach` | The extent of that box stays below `aff_reach` of any bound on the tile's extents, wherever the tile starts. |
| `Sched/Tree.lean` | `run_perm` | The leaves of a tree over all seven nodes hold every point of the root box once. |
| `Sched/Ghost.lean` | `ghost_sound` | Whatever the run-time extents are, the box a leaf computes satisfies one of the ghost states (`cap`, `dyn`, `part`, `red`) the checks saw at that leaf. |
| `Sched/Ghost.lean` | `leaf_bounded` | A bound on `cap` at the leaves bounds every run-time tile, so a buffer sized by `cap` holds it. |
| `Sched/Exact.lean` | `Op.exec_eq_leaf` | A tree whose checks pass leaves in the output buffer what one `Leaf` on its box leaves there. |
| `Sched/Exact.lean` | `Op.exec_spec` | From an uninitialized buffer, such a tree computes `post` of each cell's left fold. |
| `Sched/View.lean` | `view_sound` | Below any `Split`, `Pack` and `Stage` whose checks pass, a leaf's views hold data and supply the values their slots specify on its box. |
| `Sched/Machine.lean` | `MOp.entry_spec` | The main theorem: the machine over views, entered as `run_levels_at` enters it, computes the specification. |
| `Sched/Machine.lean` | `MOp.fill_spec` | A producer that is itself such a run meets the premise a `Stage` has of it. |
| `Sched/Rfactor.lean` | `Merge.rfactor`, `Op.redLanes_eq` | Folding `w` lanes from `init` and merging them is the left fold when the merge makes the accumulators a commutative monoid. |
| `Sched/Alloc.lean` | `stage_fits`, `pack_fits` | What a `Stage` or `Pack` fills at run time stays within the static count it checks against its `Alloc`. |
| `Sched/Need.lean` | `need_fits` | The storage `levels_need` computes passes that check at every `Pack` and `Stage` that takes it. |
| `Sched/Leaf.lean` | `Op.exec_paths` | A tree whose leaves take the scratch tile of a run-time edge or the lanes of a `Reduce` computes what it computes with plain leaves. |
| `Sched/Leaf.lean` | `AffE.flat_read` | A buffer shifted to where the tile's corner maps is read at the linear part of an affine row. |
| `Sched/Par.lean` | `Op.par_tiles` | The tiles of a `Grid` or `Threads` split of an output loop leave the same output in whatever order they run. |
| `Sched/Par.lean` | `strided_perm` | The work-items of a `Strided` split visit every element once. |
| `Sched/Store.lean` | `trace_safe`, `alloc_safe` | Every read of an `Alloc`'s storage sees the write it expects: the tiles of a `Split` write it in turn, and no node below a writer writes it again. |
| `Sched/Store.lean` | `life_safe` | No leaf uses a buffer of a `Pack` or `Stage` that is not live, and each of them frees its buffer after its last use, in the order of allocation. |
| `Sched/Store.lean` | `edge_panel` | Under `echeck`, an edge tile reads lanes of one panel, which lies below the count `levels_pack_one` allocates. |
| `Sched/Bounds.lean` | `echeck_sound`, `reads_in_home` | Under `echeck`, every position a leaf reads, an edge tile's beyond its extent included, lies in a panel of the box its view was built for. |
| `Sched/HSafe.lean` | `htrace_safe`, `root_safe` | The memory-safety theorem: a tree whose checks pass runs on a heap with addresses without a fault, whatever the run-time extents are, and leaves the heap as it was. |
| `Sched/Corpus.lean` | 19 examples | The model rejects each tree that `lit/tensor/pe_cpu_sched_check.mim` has `tensor.mim` reject, one per `SchedTree:` message, and accepts the trees accepted next to them. |

No theorem uses `sorry`, and none depends on an axiom beyond `propext`, `Classical.choice` and `Quot.sound`.

`Op.exec_spec` has two cases.
A combiner without a merge (`lanes = 0`) passes the checks only where no leaf sees a reduction loop after the first one cut, and then every cell's additions keep their order: the result is exact for floating point.
Any other tree needs additions whose order does not matter, which `Merge.comm_f` derives from a merge.

The checks are functions of the ghost state alone: `check` for the axis registers and `vcheck` for the views.
`ghost_sound` and `view_sound` are what makes evaluating them at specialization time enough.

## The heap

`Sched/Heap.lean` has a heap of blocks with an address, a capacity, the shape they currently hold data in, and a tag for the fill they hold.
`Sched/HMachine.lean` gives, for one slot, the events a run has on it: the leaves read every position of their tile, an edge tile up to the `cap` of its output loops; a `Pack` reads what it packs, allocates and fills its panels, and frees them after its body; a `Stage` does the same with its box; an `Alloc` holds the storage that the nodes below fill in turn.
A run is safe if its events replay without a fault: no read out of the shape of its block, no use of a freed block, no read of a block that another fill has overwritten, no fill beyond a capacity.

`htrace_safe` proves this from `hcheck`, which only looks at the static part of the state.
An edge tile reads up to `cap` only when no slot reads its own input buffer any more, as in `edge_tile`.
The machine is executable, and `Sched/HTest.lean` runs it: a tree that packs a run-time loop in panels narrower than the `cap` of its edge tile reads out of bounds there, and `hcheck` rejects it.
`tensor.mim` used to accept such a tree; the two checks of `hcheck` that it lacked are now in `exec_leaf` and `exec_pack`, with their rejections in `lit/tensor/pe_cpu_sched_check.mim`.

## Checking the trees the compiler builds

`mim -VVVV` logs every schedule tree `LowerToMem` builds together with the state `run_levels` starts it in.
`sched_check` reads such a log and runs `Tree.wf`, `scheck` and `vcheck` on each tree:

```sh
lake build
./check_lit.py ../.. ../../build/bin/mim
```

`check_lit.py` compiles every lit test that loads `tensor` once per set of `-X` options of its `RUN` lines.
On this branch these are 160 compilations with 340 trees, and the model accepts all of them.
266 of them are the trees at the root, and 74 are the trees of fused producers.
A producer's tree depends on the box its `Stage` asks for, which only partial evaluation knows.
At `-VVVV`, `LowerToMem` therefore wraps the producer's tree in `tensor.sched_show`, whose normalizer logs the tree once that box is static.

`sched_check` also runs `echeck` and, per slot, `hcheck`, and reports a tree that fails them without rejecting it.
All 340 trees pass both.
`echeck` asks more than `hcheck`: it takes every edge tile to run up to `cap`, whether or not a slot still reads its own buffer.
The log has no access maps, so `hcheck` runs without the rows of a slot and bounds no `Stage` by an `Alloc` there.

It rejects trees mutated in the log, e.g. a `Loop` turned `Fixed` over a run-time axis, a flipped `dep` of a `Pack`, a `Stage` removed above its producer's readers, or a `Split` that cuts panels.

## How the model differs from `tensor.mim`

- Loops and symbols are naturals and a box is a list, where `tensor.mim` has `Idx Rn` and arrays of length `Rn`.
- A run is the list of boxes its leaves compute, and one `tiles` yields the tiles of all three split modes.
  The modes only differ in the ghost state their body runs in, which `atiles` models after `split_axis`.
- A divisor or modulus of an `AffE` is positive (`AffE.WF`); `tensor.mim` does not check this.
- The output cell of a point is its first `ro` coordinates and the loops start at `origin = 0`, where `tensor.mim` has an injective `map_out` and a producer's origin.
- A buffer is a value: a panel or a staged box is a function, and what a `Pack` does not fill holds an arbitrary `junk`.
- A static reduction extent is positive; a run-time one of 0 takes the path of `run_levels_at` that fills the output with `init`.

## What is missing

- Layout: a block has a shape, not a row-major array of cells, so that a shape of `n` elements fits a capacity of `n` is where the model stops.
  The heap holds no values either; that a read returns the specified value is `view_sound`, and that it reads the fill it expects is the tag.
- Memory outside the slots: the output buffer, the scratch tile of an edge leaf and the lanes of a `Reduce` have no events.
  A producer's own run is a run of the same machine and is not nested in its consumer's.
- `need_fits` asks that no second `Alloc` of the slot lies below the first.
  In `tensor.mim` the outer one counts nothing for the users below an inner one, and an inner one without a static bound hands them the outer storage, so their check may then reject the tree.
- The correspondence between the model and `tensor.mim` itself.
  It rests on reading both, on `sched_check` agreeing with the compiler on the corpus, and on `Sched/Corpus.lean`; nothing compares what the two compute.
- `fill_spec` takes the checks of a producer's run for every box as a premise; the log holds the boxes of the corpus.
- The GPU interpreters of the `pe-tree` branch: only the tiles of their split modes are modelled, not shared memory, the ring of a `Piped` split, or an order of tiles finer than whole tiles.
