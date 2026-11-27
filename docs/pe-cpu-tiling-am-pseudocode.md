# The `run_levels` abstract machine: Python-style pseudocode

This document restates the interpreter `run_levels_at` of `src/mim/plug/tensor/tensor.mim` in Python-style pseudocode (as of commit `ad1edc5d2`).
It is meant as a specification to read next to the source, from the point of view of an abstract machine: every instruction has one `exec_*` function, and `eval` only dispatches.
The pseudocode leaves out type parameters, the threading of the memory token step by step, and some of the messages of `refly.check`, but it keeps every control structure and every check.

---

## 0. Conventions: compile time and run time

The interpreter is an ordinary Mim program that partial evaluation (PE) runs at compile time.
The values in the pseudocode therefore fall into two classes:

| Notation | Meaning | Examples |
| :--- | :--- | :--- |
| a plain Python value | **static**: PE computes it at compile time, and it does not occur in the residual program | the nodes of the schedule tree, `ts`, `cap`, `dyn`, `part`, `red`, `vdim`, `flat` |
| `Dyn[...]` | **dynamic**: only known at run time, PE leaves the code that computes it | the `size` of a run-time shape, the loop variable `t`, `lo` inside a `Loop` |
| `emit_*` | emits residual code: the loops, allocations and branches that stay in the IR after PE | `emit_for`, `emit_if`, `btensor.map_reduce_post` |

As long as the tree is closed (`core.pe.is_closed`), the `match` of `eval` is decided at compile time, and the dispatch, the environment and the continuations of the machine are all erased.
What remains is the loop nest and the buffer operations that the `emit_*` calls produce.

---

## 1. Instructions and machine state

### 1.1 Instructions (`SchedTree`)

```python
class SplitMode(Enum):
    Fixed = 0   # static extent; the tail is peeled statically
    Loop  = 1   # counts the tiles at run time; the body is unfolded once
    Peel  = 2   # like Fixed on a static extent; on a run-time one, a loop over whole tiles and one edge tile that may be empty

@dataclass
class Leaf:   vdim: int                                   # emits the micro-kernel; vdim is the vectorized loop (== Rn for none)
@dataclass
class Split:  d: int; ts: int; mode: SplitMode; body: Tree # cuts axis d into tiles of ts, a serial affine.For
@dataclass
class Reduce: d: int; w: int; body: Tree                   # reduction axis d keeps w partial accumulators that merge combines
@dataclass
class Pack:   d: int; w: int; slot: int; dep: list[bool]; body: Tree  # packs the input slot along d into BLIS panels of width w
@dataclass
class Stage:  slot: int; body: Tree                        # materializes the producer on the box that body reads
@dataclass
class Alloc:  slot: int; body: Tree                        # allocates once the storage of the outermost Pack/Stage of slot in body (store_at)
@dataclass
class Switch: d: int; c: int; small: Tree; big: Tree       # runs small if the extent of axis d is below c, and big otherwise

Tree = Leaf | Split | Reduce | Pack | Stage | Alloc | Switch
```

Derived combinators are not part of the core ADT:

```python
def sched_par(sizes, body):        # a Fixed split of several axes, the first one outermost
    for d in reversed(range(Rn)):
        if sizes[d] != 0: body = Split(d, sizes[d], Fixed, body)
    return body
def sched_seq(d, ts, body):  return Split(d, ts, Loop, body)
def sched_peel(d, ts, body): return Split(d, ts, Peel, body)
def sched_levels(levels, vdim):    # an IREE-style lowering_config: one sched_par per level
    return foldr(sched_par, levels, Leaf(vdim))
```

Every edit of a tree is an instance of `sched_fold`, a bottom-up catamorphism: `sched_unpack`, `sched_retile`, `sched_at_leaves`, `sched_with_levels`, `sched_size`.

### 1.2 Environment: axis registers and input views

```python
@dataclass
class Axis:                 # one per loop axis, Rn in all
    lo:   Dyn[int]          # where the current sub-box starts on this axis, relative to origin
    size: Dyn[int]          # the extent of the current sub-box; static if dyn is false
    cap:  int               # a static upper bound that types scratch tiles and panels
    dyn:  bool              # whether size is only known at run time
    part: bool              # whether a Split may have cut the axis into tiles smaller than its whole extent
    red:  int               # the partial accumulators a Reduce asks for, or 0

@dataclass
class View:                 # LevelsView: an input slot as the loops below the current node see it
    r: int; s: list[int]    # rank and shape of the current data buffer (the rank grows by 1 when packed)
    unit: list[int]         # the elements one step along each axis stands for (the panel width w on a packed axis)
    q: list[AffIndex]       # the origin of the view (counted in panels once packed)
    at: Callable            # (q, o) -> buffer coordinates: the read map from loop coordinates to data coordinates
    data: Buf               # the buffer read now: the original one, a panel, or the box of a Stage
    aff: Aff                # the original access map, as data, for computing boxes
    prod: Prod              # Plain (holds its data) or Fill(make) (a producer that needs a Stage)
    reads: list[bool]       # the loops the slot reads along
    flat: bool              # the original buffer with an affine map: a Leaf may shift the buffer to the corner of its tile and read at the linear part
    sn: int                 # the elements of the storage an Alloc above provides, or 0
    store: Buf              # that one-dimensional storage; the next Pack/Stage takes it as its panel or box
    own: bool               # whether data is still the slot's own input buffer, rather than a panel or box sized by cap
```

`own` decides how far an edge tile may read: a slot's own buffer ends at its extent, while panels and materialized boxes are sized by `cap` (see §8.3).

### 1.3 Store and continuation

```python
@dataclass
class State:
    axes:  list[Axis]       # the environment E, part one
    views: list[View]       # the environment E, part two
    tok:   MemToken         # the store S: the memory token, threaded explicitly
    dst:   Buf              # the store S: the output buffer that accumulates (Ro, So)

Body = Callable[[State], tuple[MemToken, Buf]]   # the continuation K
```

`eval(tree)` compiles a tree into a `Body`.
Every `exec_X(fields..., k)` takes the compiled continuation `k` of its subtree and returns a new `Body`.

---

## 2. Entry

```python
def run_levels_at(So, Sr, origin,                  # output shape, loop extents, loop origin (for a producer's box)
                  f, init, post,                   # combiner, identity, per-cell post-processing
                  merge, lanes, finish,            # merges partial results; lanes == 0 means the combiner has no merge; finish: run epi
                  epi, eager,                      # epilogue (may take outside inputs, e.g. a bias); eager: a Leaf may apply post itself
                  drain, drains,                   # consumers that read each finished tile (Section 8.5); drains: there are any
                  map_out, maps, prods,            # output map, the Aff read map of every input, the Prod of every input
                  sched, cap0, dyn0,               # schedule tree, static bound per axis, whether each axis is run-time
                  tok, dst, xs0):                  # the caller provides the output buffer
    views0 = [View(r=Rs[i], s=Ss[i], unit=[1]*Rn, q=origin,
                   at=levels_plain(maps[i]), data=xs0[i], aff=maps[i], prod=prods[i],
                   reads=aff_reads(maps[i]), flat=aff_linear(maps[i]), sn=0, store=None, own=True)
              for i in range(N)]
    axes0 = [Axis(lo=0, size=Sr[d], cap=cap0[d], dyn=dyn0[d], part=False, red=0) for d in range(Rn)]
    run = lambda tok: eval(sched)(State(axes0, views0, tok, dst))

    red_dyn = any(dyn0[d] for d in range(Ro, Rn))
    if not red_dyn:
        return run(tok)

    # A reduction axis has a run-time extent: at extent 0 the tree does not run and no Leaf writes the output,
    # so the cells take init directly, and epi maps them
    def empty(tok):
        tok = emit_buffer_set(dst, init, tok)
        tok, dst = epi(tok, dst, identity, So) if finish else (tok, dst)
        return (drain(tok, dst, origin, Sr, cap0, dyn0) if drains else tok), dst
    return emit_if(any(Sr[d] == 0 for d in range(Ro, Rn)), then=empty, else_=run)
```

The public wrappers only fix some of the parameters:

| Entry | Fixed parameters |
| :--- | :--- |
| `run_levels` | every `prods = Plain`, `post = id`, `lanes = 0`, no epilogue; allocates the output itself |
| `run_levels_into` | `post = id`, `lanes = 0`, `epi = levels_no_epilogue`, `eager = True`; the caller provides the output buffer |
| `run_levels_with` | allocates the output and calls `run_levels_into` |
| `run_levels_post` | `origin = 0`, everything else passed through, allocates the output itself (the one `lower_to_mem` in C++ uses) |

---

## 3. Dispatch

```python
def eval(node: Tree) -> Body:            # PE only unfolds it for a closed node
    match node:
        case Leaf(vdim):                 return lambda st: exec_leaf(vdim, st)
        case Split(d, ts, mode, body):   return exec_split(d, ts, mode, eval(body))
        case Reduce(d, w, body):         return exec_reduce(d, w, eval(body))
        case Pack(d, w, slot, dep, body):return exec_pack(d, w, slot, dep, eval(body))
        case Stage(slot, body):          return exec_stage(slot, eval(body))
        case Alloc(slot, body):          return exec_alloc(slot, body, eval(body))   # also looks at the subtree itself, for the capacity
        case Switch(d, c, small, big):   return exec_switch(d, c, eval(small), eval(big))
```

Every instruction changes only part of the state:

| Instruction | Changes | Leaves alone |
| :--- | :--- | :--- |
| `Split` | `axes[d]` (lo/size/cap/dyn/part), the `q` of every view | the data of `views`, `dst` |
| `Reduce` | `axes[d].red` | everything else |
| `Pack` | `views[slot]` (replaced by the panel) | `axes`, `dst` |
| `Stage` | `views[slot]` (replaced by the materialized box) | `axes`, `dst` |
| `Alloc` | `views[slot].sn/store` | everything else |
| `Switch` | `axes[d].cap` (tightened in the small branch) | everything else |
| `Leaf` | `dst`, `tok` | `axes`, `views` |

---

## 4. `Split`: cutting the iteration window

```python
def exec_split(d, ts, mode, k):
    check(d < Rn,  "a Split needs an axis below Rn")
    check(ts > 0,  "a Split needs a positive tile size")
    rt_tail, peel = {Fixed: (False, False), Loop: (True, False), Peel: (False, True)}[mode]
    return split_axis(rt_tail, peel, d, ts, k)

def split_axis(rt_tail, peel, d, ts, k):
    def body(st):
        a = st.axes[d]
        n = a.size
        check(not a.dyn or rt_tail or peel, "a Fixed Split needs an axis with a static extent")
        full, tail = n // ts, n % ts                    # both are Dyn if a.dyn
        bound = min(a.cap, ts)
        part  = a.part or a.dyn or ts < n

        def tile(k, t, ext, cap, dyn, tok, dst, xs):    # runs tile t
            views2 = [v._replace(q=shift_q(v, d, t, ts), data=xs[i]) for i, v in enumerate(st.views)]
            # a packed axis must be cut into whole panels: ts % v.unit[d] == 0,
            # or else "a Split must cut a packed axis into whole panels"
            axes2 = st.axes.copy()
            axes2[d] = Axis(lo=a.lo + t * ts, size=ext, cap=cap, dyn=dyn, part=part, red=a.red)
            return k(State(axes2, views2, tok, dst))

        def wholes(k, after):                           # whole tiles: a static extent without any skips the loop
            if not a.dyn and full == 0: return after(st.tok, st.dst)
            return emit_for(full, lambda t, tok, dst: tile(k, t, ts, ts, False, tok, dst, ...), after)

        def tail_static(k):  return lambda tok, dst: tile(k, full, tail, tail, False, tok, dst, ...) if tail else (tok, dst)
        def tail_branch(k):  return lambda tok, dst: emit_if(tail != 0,
                                                             then=lambda: tile(k, full, tail, bound, True, tok, dst, ...),
                                                             else_=lambda: (tok, dst))
        def counted(k):                                 # counts the tiles at run time: full + (tail != 0) of them
            return emit_for(full + (tail != 0),
                            lambda t, tok, dst: tile(k, t, min(n - t*ts, ts), bound, True, tok, dst, ...))

        # a run-time extent counts its tiles unless peeled; a static one only for a Loop with whole tiles and a tail
        count = (not peel) if a.dyn else (rt_tail and full != 0 and tail != 0)
        if count:      return counted(k)
        if not a.dyn:  return wholes(k, tail_static(k))   # Fixed / static Peel / static Loop
        return wholes(k, tail_branch(k))                  # run-time Peel
    return body
```

Notes:
- On a static extent, `Fixed` unfolds the tail into a second copy of static code, so the main tiles and the tail both run at full speed with `dyn = False`.
- `Loop` unfolds the subtree only once; the price is that the subtree sees a `Dyn` `size` and has to type its scratch by `cap = bound`.
- On a run-time extent, `Peel` gets both: the loop over whole tiles has the static `ts`, and only one edge tile is `Dyn`.

---

## 5. `Reduce`: splitting a reduction

```python
def exec_reduce(d, w, k):
    check(Ro <= d < Rn, "a Reduce needs a reduction loop")
    check(w > 1,        "a Reduce needs at least two partial results")
    def body(st):
        axes2 = st.axes.copy()
        axes2[d] = st.axes[d]._replace(red=w)
        return k(st._replace(axes=axes2))
    return body
```

`Reduce` only records the request in the axis register; `red` reaches the tiles below through `Split` and `Switch`, and the `Leaf` carries it out (see §8.4).
The reason is that the type `SchedTree (N, Rn)` fixes the number of loops, so a node cannot give its subtree another axis for the lanes; the lane axis only exists in the local loops a Leaf emits.
Whether there is a merge is a property of the combiner and not of the node, so that check sits in the Leaf as well (checking it here would fold to a false alarm inside the body of `eval` for an instance with `lanes == 0`).

---

## 6. `Pack`: replacing an input by BLIS panels

```python
def exec_pack(d, w, slot, dep, k):
    def body(st):
        check(d < Rn and w > 0, "a Pack needs an axis below Rn and a positive width")
        v = st.views[slot]
        check(dep == v.reads,              "a Pack's dep must be the loops its slot reads")
        check(v.reads[d],                  "a Pack needs a loop its slot reads")
        check(ready(v.prod),               "a Pack reads a producer slot that no Stage above it computes")
        check(all(st.axes[e].cap < UNBOUNDED for e in range(Rn) if dep[e] or e == d),
              "a Pack sizes its panel by a run-time extent that no Split bounds")

        # the static cap gives the shape of the panels: Rn dimensions and one for the lanes;
        # a run-time extent gets a spare panel for an empty edge tile
        tok, sp, panel, to = levels_pack_one(v, d, w, dep, sizes(st), caps(st), st.axes[d].dyn, v.sn, v.store, st.tok)
        #   with the storage of an Alloc: check(prod(sp) <= v.sn, "a Pack needs more than its Alloc holds"), panel = buffer.view(sp, v.store)
        #   otherwise: emit alloc panel
        #   then fill the whole panels; fill the tail panel (lanes = tail)
        v2 = View(r=Rn + 1, s=sp, unit=[w if e == d else 1 for e in range(Rn)], q=[0]*Rn,
                  at=to, data=panel, aff=NONE, prod=Plain, reads=v.reads, flat=False, sn=0, store=None, own=False)
        tok, dst = k(st._replace(views=set_at(st.views, slot, v2), tok=tok))
        return (tok if v.sn else emit_free(panel, tok)), dst
    return body
```

Panel coordinates: `to(q, o)[k]` is `o[d] % w` for the lane, `q[d] + o[d] // w` for the panel number, `q[k] + o[k]` where `dep[k]` holds, and `0` where the slot does not depend on the loop.

---

## 7. `Stage`: materializing a box of a producer on demand

```python
def exec_stage(slot, k):
    def body(st):
        v = st.views[slot]
        match v.prod:
            case Plain():                       # holds its data: nothing to do
                return k(st)
            case Fill(make):
                at = [origin[d] + st.axes[d].lo for d in range(Rn)]
                blo, bext, bcap = aff_box(v.aff, at, sizes(st), caps(st))   # §10
                linear = [aff_span(row, v.aff.vals, at, sizes(st)).affine for row in v.aff.rows]
                # a row with floordiv/ceildiv/mod: the extent of the box depends on where the tile starts,
                # and is Dyn if a run-time axis is involved
                bdyn = [not linear[i] or any(st.axes[d].dyn and mentions(row_i, d) for d in range(Rn))
                        for i, row_i in enumerate(v.aff.rows)]
                if v.sn:                                        # the storage of an Alloc above
                    check(prod(bcap) <= v.sn, "a Stage needs more than its Alloc holds")
                    tok, t = st.tok, buffer_view(bext, v.store)
                else:
                    tok, t = emit_alloc(bext, st.tok)
                tok, t = make(blo, bext, bcap, bdyn)(tok, t)    # only fills; recursive: the producer is a run_levels_into of its own
                def rel(q, o):                  # reads the box: an affine row at its linear part, any other row at its value less the start of the box
                    x = q + o; y = at + x
                    return [aff_rel(row, x).value if linear[i]
                            else aff_eval(row, v.aff.vals, y) - blo[i]
                            for i, row in enumerate(v.aff.rows)]
                v2 = View(r=v.r, s=bext, unit=[1]*Rn, q=[0]*Rn, at=rel, data=t,
                          aff=v.aff, prod=Plain, reads=v.reads, flat=False, sn=0, store=None, own=False)
                tok, dst = k(st._replace(views=set_at(st.views, slot, v2), tok=tok))
                return (tok if v.sn else emit_free(t, tok)), dst
    return body
```

`make` comes from `fill_of` (a map-reduce producer that calls `run_levels_at` again with a tree of its own policy and `origin = lo`; its output map may permute the output loops, and it may carry an epilogue without inputs), from `gather_fill`, or from `pad_fill` (a constant pad: set the box to the value, then copy the part of the input that the box overlaps).
`make` only fills the buffer its caller gives it; allocating is up to the `Stage` or to an `Alloc` above, which is the separation of store_at from compute_at.
This is fusion that starts at the root and pulls producers on demand: a producer only computes the box that this tile of its consumer reads.

---

## 8. `Leaf`: the micro-kernel

### 8.1 Checks and read maps

```python
def exec_leaf(vdim, st):
    check(vdim <= Rn, "a Leaf vectorizes a loop below Rn, or none with Rn")
    # without a merge only the first reduction axis may be cut, since anything else reorders the combination
    ordered = not any(st.axes[d].part for d in range(Ro + 1, Rn))
    check(lanes > 0 or ordered, "only the first reduction loop may be split when the combiner has no merge")

    axes, vs, tok, dst = st
    lanes_v = axes[vdim].red if vdim < Rn else 0
    check(all(axes[d].red == 0 for d in range(Rn) if d != vdim),
          "a Reduce must split the loop its Leaf vectorizes")

    out = lambda o: map_out([axes[d].lo + o[d] for d in range(Rn)])   # output coordinates (global)
    corner = [origin[d] + axes[d].lo for d in range(Rn)]

    # flat: the original buffer with an affine map; the buffer is shifted to where the corner of the tile maps,
    # and the tile addresses it at the linear part only
    ins_m = [ (lambda o: [aff_rel(row, o).value for row in v.aff.rows]) if v.flat
              else (lambda o: v.at(v.q, o))
              for v in vs ]
    datas = [ buffer.shift(v.data, [aff_span(row, v.aff.vals, corner, [1]*Rn).lo for row in v.aff.rows])
              if v.flat else v.data
              for v in vs ]
    check(all(ready(v.prod) for v in vs), "a Leaf reads a producer slot that no Stage above it computes")

    fresh = all(axes[d].lo == 0 for d in range(Ro, Rn))          # the first tile of the reduction: starts from init instead of reading dst
    whole = eager and not any(axes[d].part for d in range(Ro, Rn))  # covers the whole reduction: the Leaf may apply post itself
```

### 8.2 Emitting the kernel

```python
    def kernel(vec, S, c, at, sz, tok):       # vec=False: the extent is a run-time value, so nothing is vectorized
        return btensor.map_reduce_post(
            shape=S, iters=sz, sched=mk_sched(vec=Rn if (not vec or vdim < Ro) else vdim),
            comb=f, acc=(S, at, c, fresh, init), post=(post if whole else id),
            out=at, ins=ins_m, bufs=datas)(tok, c)
```

### 8.3 The edge tile of a run-time extent

```python
    def edge_tile():
        # the size of an output axis is Dyn: compute in a scratch tile sized by cap, and copy the valid part back.
        # The kernel only runs up to cap when panels or boxes sized by cap hold every slot;
        # as soon as one slot still reads its own buffer, which ends at the extent, it runs over the valid part, unvectorized.
        owned   = any(v.own for v in vs)
        full_sz = [axes[d].cap if (d < Ro and not owned) else axes[d].size for d in range(Rn)]
        Sc  = [axes[d].cap  for d in range(Ro)]
        ext = [axes[d].size for d in range(Ro)]
        tok, sc = emit_alloc(Sc, tok)
        tok, sc = emit_copy(dst -> sc, ext, offset=lo, tok)
        tok, sc = kernel(not owned, Sc, sc, loops, full_sz, tok)   # the scratch tile is indexed by the loop coordinates
        tok, dst2 = emit_copy(sc -> dst, ext, offset=lo, tok)
        return emit_free(sc, tok), dst2
```

### 8.4 The rfactor a `Reduce` asks for

```python
    def rfactor():
        w = lanes_v
        n = axes[vdim].size
        full, tail = n // w, n % w
        # the buffer p of partial accumulators: the Ro output dimensions and one for the lanes, of width w
        tok, p = emit_alloc(Sp := [*sizes[:Ro], w], tok)
        # the iteration space has Rn + 1 dimensions: the lane is dimension Ro (an output dimension), vdim becomes the
        # group number, and the original coordinate is base + group*w + lane
        def part(base, width, groups, seed, tok):
            return btensor.map_reduce_post(iters=[*sizes[:Ro], width, ..., groups, ...],
                                           sched=mk_sched(vec=Ro),     # vectorizes the lane dimension
                                           comb=f, acc=(p, seed, init),
                                           ins=[lambda c: ins_m[i](back(base, c)) for i in range(N)])(tok, p)
        tok = part(0, w, full, seed=True, tok)  if full else part(0, tail, 1, seed=True, tok)
        if full and tail: tok = part(full * w, tail, 1, seed=False, tok)  # the tail goes into the first tail lanes
        used = w if full else tail
        # merge: reduce along the lane dimension into dst with merge; apply post as well if whole
        tok, dst2 = btensor.map_reduce_post(iters=[*sizes[:Ro], used], comb=merge,
                                            acc=(dst, fresh, init), post=(post if whole else id))(tok, p)
        return emit_free(p, tok), dst2
```

### 8.5 Picking a path, and the epilogue

```python
    red_v = lanes_v > 1
    check(not red_v or lanes > 0, "a Reduce needs a combiner with a merge")
    edge  = blk(vdim) or blk(vdim + 1)                   # the size of the vectorized axis or of the next one is Dyn
    check(not edge or all(axes[d].cap < UNBOUNDED for d in range(Ro)),
          "a Leaf computes an edge tile of a run-time extent that no Split bounds")
    # an edge tile that runs up to cap reads up to cap along a run-time output axis: a slot that reads that axis must be
    # packed along it, in panels at least as wide as cap, so that the reads stay in the panel the tile starts in
    owned = any(v.own for v in vs)
    wide  = all(v.unit[d] >= axes[d].cap
                for v in vs for d in range(Ro) if axes[d].dyn and v.reads[d])
    check(not edge or owned or wide,
          "a Leaf computes an edge tile wider than a panel of a slot it reads along a run-time loop")

    if red_v and not blk(vdim): tok, dst = rfactor()
    elif edge:                  tok, dst = edge_tile()
    else:                       tok, dst = kernel(True, So, dst, out, sizes, tok)

    # the Leaf that ends the reduction of its cells runs epi (e.g. + bias, activation) while they are still in cache,
    # and then hands them to every consumer in drain
    mapped = finish and not whole
    if mapped or drains:
        last = all(axes[d].lo + axes[d].size == Sr[d] for d in range(Ro, Rn))   # static or Dyn
        def finished(tok, dst):
            if mapped: tok, dst = epi(tok, dst, cells, ext)
            if drains: tok = drain(tok, dst, corner, ext, cap, dyn)   # the tile's output loops, as Nat
            return tok, dst
        tok, dst = emit_if(last, then=lambda: finished(tok, dst), else_=lambda: (tok, dst))
    return tok, dst
```

A `drain` is `levels_sink` for one consumer, or `levels_drain_both` of several; it is off unless `-X tensor:pe-drain=1`.
`levels_sink` runs the consumer's own `run_levels_at` over the finished tile, with `origin` at the tile's corner and the tree `sink_tree` gives: each loop of the consumer tied to an output loop of the producer runs over the tile along it, every other loop over its whole extent.
A leaf seeds a cell with `init` only where its reduction loops start at 0 in global coordinates (`origin + lo`), so the tiles that cut a tied reduction loop fold into what earlier tiles left.
Three checks make a wrong sink fail instead of computing wrong values:

```python
    check(all(not tied[d] for d in loops if reads(f_aff, d)),
          "a sink reads its tile along a loop that runs over its whole extent")
    blo, _, bcap = aff_box(f_aff, origin, size, bound)
    check(all(blo[e] == corner[dim_loop[e]] and bcap[e] == cap[dim_loop[e]] for e in range(Ro)),
          "a sink reads other cells than the tile it is given")
    check(all(lanes > 0 for d in loops if tied[d] and d > Rc),
          "a sink cuts a reduction loop after the first without a merge")
```

The C++ side (`LowerToMem::sinks_of`) only picks the consumers and ties their loops; it may be wrong without making the result wrong.

---

## 8.6 `Alloc`: separating where data is stored from where it is computed

```python
def exec_alloc(slot, tree, k):
    def body(st):
        v = st.views[slot]
        box = lambda caps: prod(aff_reach(row, caps) + 1 for row in v.aff.rows)   # the size of a Stage's box for given bounds
        n = levels_need(slot, box, fills=not ready(v.prod), caps=caps(st), dyn=dyns(st), t=tree)
        if n >= UNBOUNDED: n = 0           # a run-time extent without a static bound: every Pack/Stage allocates its own again
        if n == 0: return k(st)
        tok, store = emit_alloc((n,), st.tok)
        v2 = v._replace(sn=n, store=store)
        tok, dst = k(st._replace(views=set_at(st.views, slot, v2), tok=tok))
        return emit_free(store, tok), dst
    return body

def levels_need(want, box, fills, caps, dyn, t):   # a static abstract interpretation of the subtree that only follows cap
    match t:
        case Leaf():                return 0
        case Split(d, ts, _, b):    return levels_need(..., caps with d -> (ts if dyn[d] else min(caps[d], ts)), b)
        case Reduce(_, _, b):       return levels_need(..., caps, b)
        case Pack(d, w, s, dep, b):
            if s != want:           return levels_need(..., caps, b)
            count = ceil(caps[d] / w) + 1   # always one spare panel
            return w * count * prod(caps[e] for e in range(Rn) if e != d and dep[e])
        case Stage(s, b):
            if s != want or not fills: return levels_need(..., caps, b)
            return box(caps)
        case Alloc(s, b):           return 0 if s == want else levels_need(..., caps, b)
        case Switch(d, c, sm, bg):  return max(levels_need(..., caps with d -> min(caps[d], c - 1), sm),
                                               levels_need(..., caps, bg))
```

- The capacity is one static abstract interpretation of the subtree: it only tightens `cap` along the path and takes the maximum over the outermost users; the products saturate at `levels_unbounded`.
- Only the outermost Pack/Stage use the storage; they hand their subtree a view with `sn = 0`, so nested users of the same slot allocate their own and do not overwrite each other.
- The tiles of a Split use the same storage in turn: the Pack/Stage of each tile fills again what that tile reads.
- Policy: `cpu_tree` only adds an `Alloc` at the root for a fused slot (covering the box of its Stage) and leaves the panels alone; `generic_tree` adds one per fused slot outside the band loop.
  Hoisting the panels to the root measured no faster (glibc reuses allocations of one size well), and at 512³ clang then spills one more register in the micro-kernel (Zen 5), so the default policy does not do it; a hand-written tree still can.

## 9. `Switch`: dispatching between versions

```python
def exec_switch(d, c, small, big):
    check(d < Rn, "a Switch needs an axis below Rn")
    check(c > 1,  "a Switch needs a threshold above 1")
    def body(st):
        a  = st.axes[d]
        lt = st with axes[d].cap = min(a.cap, c - 1)     # the small branch may size its scratch by c - 1
        if not a.dyn:                                    # static: picks a subtree at compile time
            return small(lt) if a.size < c else big(st)
        return emit_if(a.size < c, then=lambda: small(lt), else_=lambda: big(st))
    return body
```

Example: the outer-product register tile of `generic_tree` uses `Switch` to cut an edge that is not a power of two into power-of-two widths, level by level:

```python
def lane_tree(vl, last, leaf):          # T(w) = Split(last, w, Switch(last, w, T(w/2), leaf))
    t = leaf
    for w in [vl // 16, vl // 8, vl // 4, vl // 2, vl]:
        if w > 1: t = Split(last, w, Fixed, Switch(last, w, t, leaf))
    return t
# 28 = 16 + 12 → 12 < 16 → 8 + 4: every piece is a power of two, which LLVM vectorizes as a whole
```

---

## 10. The access map `Aff`: affine expressions as data

```python
class AffK:  Num(c) | Var(k)            # a factor, divisor or modulus: a literal, or symbol k of vals
class AffE:  Dim(d) | Sym(k) | Lit(c) | Add(x, y) | Sub(x, y) | Mul(x, k) | FDiv(x, k) | CDiv(x, k) | Mod(x, k)
@dataclass
class Aff:   s: int; vals: list[Dyn[int]]; rows: list[AffE]   # the tree is static, the symbol values vals may be run-time values

def aff_eval(e, vals, x):   ...  # builds an affine.* index expression
def aff_rel(e, x):          ...  # the linear part without the constants, e(lo + x) - e(lo); reports "not linear" at a div or mod
def aff_span(e, vals, lo, ext) -> (affine: bool, start, width_minus_1):
    match e:
        case Dim(d):     return True, lo[d], ext[d] - 1
        case Add(x, y):  adds the intervals
        case Mul(x, c):  scales the interval by c
        case FDiv(x, c): (l, w) = span(x); return False, l // c, (l + w) // c - l // c
        case CDiv(x, c): the same, rounding up
        case Mod(x, c):  r = l % c; wraps to [0, c-1] if r + w >= c, and is [r, r+w] otherwise
        case Sub(x, y):  (l, w), (k, v) = span(x), span(y); return False, l - (k + v), (l + w - k) - (l - (k + v))
def aff_reach(e, ext):      ...  # a static bound on the width that does not depend on the start, for cap; unbounded for a run-time factor or modulus
def aff_reads(f):           ...  # the loops some row mentions
def aff_unit(f, d):         ...  # d occurs only in the last row, with coefficient 1
def aff_perm(f):            ...  # the loop each row is, and whether they are distinct ones of the first m
def aff_box(f, lo, ext, cap):    # per row: (start, width, static bound on the width)
    return [span.start], [span.width + 1], [aff_reach(row, cap) + 1]
```

Natural coefficients make the inner expression monotone, so a floordiv or ceildiv maps the ends of its interval and the box is exact; a modulus only fails to wrap when the interval fits one period.
A difference is the one row that is not monotone (a flip reads `(s - 1) - i`); its span runs from the least minuend less the greatest subtrahend, and it is not affine, so a tile reads it at its value rather than its linear part.

`tensor.map_reduce_post` carries its maps as `Aff`: `map_out: Aff(Rn, Ro)`, `maps[i]: Aff(Rn, Ris[i])`, `post_maps[j]: Aff(Ro, Rps[j])`.
An impl builds its maps as data, and there is no other way in: the rows are static trees, and every value they use, literal or run-time, sits in `vals`, so composing and evaluating them always unfolds.
Before an analysis reads a map, `aff_canon` inlines its literal values as `Num`/`Lit`, keeps the run-time ones as symbols, and drops factors and divisors of 1, so that a static bound sees every literal.
`tensor.aff_eval_map` turns data back into the function it describes, which the loop-nest lowering and the btensor operations apply.
A reduction states its monoid through `tensor.fold_of (op, h)`; `fuse_tensor` keeps that form when it inlines a producer into `h`.

---

## 11. The policy layer (outside the TCB)

A policy only builds a tree; the checks of the interpreter are what makes it legal.

```python
def cpu_tree(...):                       # contractions (GEMM and its variants): the BLIS structure
    leaf = Leaf(m_ax)
    t = peel(m_ax, mr, peel(n_ax, nr, leaf))          # register tile
    t = Pack(a, ...)(t); t = split(m_ax, mc, t)        # panels of A and the L2 tile
    t = Pack(b, ...)(t) if keep_b else t               # panels of B (decided by reuse and 4K aliasing)
    t = Split(k_ax, kc, Loop if dyn_k else Fixed, t)
    t = split(n_ax, nc, t)
    t = Stage(slot, t), placed per band where nothing is computed twice   # fused producers
    t = Switch(m_ax, 16*mr + 1, build(pack_b=True), build(pack_b=False)) if m_dyn else build(keep_b)
    return Alloc(slot, t) for each fused slot           # one allocation of the box outside all loops; panels are not hoisted

def generic_tree(...):                   # every other map-reduce: reductions, softmax, transpose, gather, attention, direct convolution
    if row:     leaf = Leaf(last)                          # vectorizes along the last output axis
    elif split: leaf = Reduce(inner, lanes, Leaf(inner))   # a contiguous reduction axis and a merge: rfactor
    else:       leaf = Leaf(Rn)
    base = sched_par(cut(256), sched_par(cut(8), leaf)) if wdim != rdim else leaf   # tiles both axes, as for a transpose
    if outer and cpu_conv:                                  # the outer-product register tile of a direct convolution
        base = Split(rows, mr, Fixed, lane_tree(vl, last, Leaf(rows)))
    staged = a Stage around base for every fused slot
    return Alloc(fused slots..., sched_par(band of axis 0, staged)) if any_fused else base
```

The C++ side (`lower_to_mem.cpp`) only does three things: it decides whether PE applies (`pe_generic_plan`, the premises of `cpu_tree`), collects the producers that can be fused (`fuse_producer`, `generic_slots`) and builds their `Prod`, and assembles the call of `run_levels_post`.

---

## 12. The checks

Every check sits in the `exec_*` of its instruction and is evaluated while PE unfolds it; a condition that is statically false is reported, and with `-X refly:strict=1` it is a compile error.

| Instruction | Checks |
| :--- | :--- |
| `Split` | the axis is below `Rn`; `ts > 0`; `Fixed` needs a static extent; a packed axis is cut into whole panels |
| `Reduce` | the axis is a reduction axis; `w > 1` |
| `Pack` | the axis is below `Rn` and `w > 0`; `dep` is what the slot reads; the slot reads the packed axis; the slot holds its data; every axis the panel involves has a finite `cap` |
| `Stage` | (none: a Plain slot passes through) |
| `Alloc` | (none: without a static bound it falls back); the `Pack`/`Stage` that uses it checks that the elements it needs do not exceed `sn` |
| `Switch` | the axis is below `Rn`; `c > 1` |
| `Leaf` | `vdim <= Rn`; without a merge only the first reduction axis is cut; the axis of a `Reduce` is `vdim`; a `Reduce` needs a merge; every slot read holds its data; the output axes of a run-time edge tile have a finite `cap`; a slot that an edge tile running up to `cap` reads along a run-time output axis is packed along it, in panels at least as wide as `cap` |

---

## 13. What is left after PE

| Part of the machine | After PE |
| :--- | :--- |
| `match node`, `eval`, the `exec_*` closures | gone |
| `cap` / `dyn` / `part` / `red` of an `Axis`, `unit` / `flat` / `prod` / `own` of a `View` | gone; they only decide which path was taken |
| `Axis.lo` / `size` | folded to constants where static, affine in the loop variable inside a `Loop` |
| `View.at`, `aff_eval` | inlined into `affine.*` index expressions, which `affine.lower_index` lowers to core arithmetic |
| `View.sn` / `store` | the one allocation of an `Alloc`; `buffer.view` lowers to a pointer bitcast |
| `emit_for` / `emit_if` / `emit_alloc` / `btensor.map_reduce_post` | the residual program: the loop nest, the panels and scratch buffers, the micro-kernel |

## Example of GEMM

Split  N (1), nc, Fixed                   // Outer N tiling (L3 level)
  Split  K (2), kc, Loop                  // K tiling (dyn_k=1 by default -> Loop, runtime tail)
    Pack   N (1), nr, slot=B, [F,T,T]     // Pack kc×nc B into nr-wide panels (omitted for small M or cache-resident B without 4KB aliasing)
      Split  M (0), mc, Fixed             // M tiling (L2 level)
        Pack   M (0), mr, slot=A, [T,F,T] // Pack mc×kc A into mr-wide panels
          Split  M (0), mr, Fixed         // Register tile along M (micro-kernel rows)
            Split  N (1), nr, Fixed       // Register tile along N (micro-kernel columns)
    		 Leaf   vdim=M (0)           // Micro-kernel accumulating along M, vectorized along N (K inside Leaf)
