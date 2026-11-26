import Sched.Heap

/-!
# The machine on the heap

The events one slot's buffers see in a run: the leaves read every position of their tile, an edge tile up to the `cap` of its output loops; a `Pack` reads what it packs and fills its panels; a `Stage` fills its box; an `Alloc` holds the storage that the nodes below fill in turn.
Address 0 is the slot's input buffer, and a node at depth `k` of the allocations allocates address `k + 1`.
The state has a static part, which the checks look at, and a part that only the run knows.
-/

namespace Sched

/-- The static part of the state: the ghost state of the slot's view and of the axes, the storage an `Alloc` above offers as its address and capacity, and the depth of the allocations. -/
structure HG where
  ready : Bool
  unit  : Vec
  reads : Nat → Bool
  vals  : Nat → Nat
  rows  : List AffE
  store : Option (Addr × Nat)
  depth : Nat
  a     : AState
  /-- Per slot, whether it still reads its own input buffer, and whether it started as a producer. -/
  own   : List Bool
  fills : List Bool

/-- What only the run knows of the slot's view: the block and the fill it reads, its origin, the index of a tile-local point, and its home box. -/
structure HD where
  loc  : Addr
  tag  : Box
  q    : Vec
  idx  : Vec → Vec → List Nat
  home : Vec
  hsz  : Vec

/-- The ghost state `split_axis` runs the tile `tl` in. -/
def atile (ts : Nat) (mode : Mode) (x : AAxis) (tl : Tile) : AAxis :=
  let part := x.part || x.dyn || decide (ts < x.cap)
  if x.dyn then
    if mode = .peel ∧ tl.2 = ts then ⟨ts, false, part, x.red⟩ else ⟨min x.cap ts, true, part, x.red⟩
  else if mode = .loop ∧ x.cap / ts ≠ 0 ∧ x.cap % ts ≠ 0 then ⟨min x.cap ts, true, part, x.red⟩
  else ⟨tl.2, false, part, x.red⟩

/-- An edge tile runs up to the `cap` of its output loops only when no slot reads its own buffer, which ends at the extent. -/
def full (g : HG) (vdim : Nat) : Bool := edgeAt g.a vdim && g.own.all (!·)

/-- The tile-local positions a leaf reads: its extents, or the `cap` of its output loops. -/
def readBox (rn ro : Nat) (full : Bool) (a : AState) (b : Box) : Box :=
  (List.range rn).map fun e => (0, if e < ro ∧ full = true then capOf a e else boxSz b e)

/-- A `Pack` gives its slot panels. -/
def ownPack (s : Nat) (g : HG) : HG := { g with own := g.own.set s false }

/-- A `Stage` gives a producer's slot a box. -/
def ownStage (s : Nat) (g : HG) : HG := { g with own := g.own.set s (g.own.getD s false && !g.fills.getD s false) }

/-- The tile-local positions a `Pack` reads: the loops its slot reads, and one point of the others. -/
def fillBox (rn d : Nat) (dep : Nat → Bool) (b : Box) : Box :=
  (List.range rn).map fun k => (0, if k = d ∨ dep k = true then boxSz b k else 1)

/-- The shape of the panels of `levels_pack_one`: per loop the panels of `d`, the `cap` of a loop the slot reads, or 1; then the lanes. -/
def packShape (rn d w : Nat) (dep : Nat → Bool) (a : AState) : List Nat :=
  ((List.range rn).map fun k =>
    if k = d then AffE.up w (capOf a d) + (if dynOf a d then 1 else 0) else if dep k = true then capOf a k else 1) ++ [w]

def packIdx (rn d w : Nat) (dep : Nat → Bool) (q o : Vec) : List Nat :=
  ((List.range rn).map fun k => if k = d then q d + o d / w else if dep k = true then q k + o k else 0) ++ [o d % w]

def packG (d w : Nat) (g : HG) : HG := { g with unit := fun e => if e = d then w else 1, store := none }

def packD (rn d w : Nat) (dep : Nat → Bool) (id : Addr) (b : Box) : HD :=
  { loc := id, tag := b, q := fun _ => 0, idx := packIdx rn d w dep, home := boxLo b, hsz := boxSz b }

def stageShape (g : HG) (b : Box) : List Nat := g.rows.map fun r => (affBox g.vals (boxLo b) (boxSz b) r).2

def stageIdx (g : HG) (b : Box) (q o : Vec) : List Nat :=
  g.rows.map fun r =>
    if (r.span g.vals (boxLo b) (boxSz b)).1 then (r.rel fun e => q e + o e).2
    else r.eval g.vals (fun e => boxLo b e + (q e + o e)) - (affBox g.vals (boxLo b) (boxSz b) r).1

def stageG (g : HG) : HG := { g with ready := true, unit := fun _ => 1, store := none }

def stageD (g : HG) (id : Addr) (b : Box) : HD :=
  { loc := id, tag := b, q := fun _ => 0, idx := stageIdx g b, home := boxLo b, hsz := boxSz b }

/-- The elements a `Stage` of the slot fills at most, for loops bounded by `caps`. -/
def stageBound (g : HG) (caps : Nat → Nat) : Nat := prod (g.rows.map fun r => r.reach caps + 1)

/-- The `cap` of a loop, and 1 beyond the loops, as `boxSz` has it. -/
def capOf1 (a : AState) (e : Nat) : Nat := match a[e]? with | some x => x.cap | none => 1

/-- The storage an `Alloc` of the slot reserves; 0 without a static bound, where each node below allocates its own. -/
def allocSize (rn slot : Nat) (g : HG) (body : Tree) : Nat :=
  let n := need rn slot (stageBound g) (!g.ready) body (capOf g.a) (dynOf g.a)
  if n < unbounded then n else 0

/-- The events of the slot's buffers in a run of the tree on the box. -/
def htrace (rn ro slot : Nat) : Tree → HG → HD → Box → List HEv
  | .leaf vdim, g, dv, b =>
    if g.ready then (points (readBox rn ro (full g vdim) g.a b)).map fun o => .read dv.loc dv.tag (dv.idx dv.q (vecOf o))
    else [.fault]
  | .split d ts mode body, g, dv, b =>
    match b[d]?, g.a[d]? with
    | some (lo, n), some x =>
      (tiles ts lo n).flatMap fun tl =>
        htrace rn ro slot body { g with a := g.a.set d (atile ts mode x tl) }
          { dv with q := fun e => if e = d then dv.q d + (tl.1 - lo) / ts * (ts / g.unit d) else dv.q e } (b.set d tl)
    | _, _ => []
  | .reduce _ _ body, g, dv, b => htrace rn ro slot body g dv b
  | .pack d w s dep body, g0, dv, b =>
    let g := ownPack s g0
    if s = slot then
      if g.ready then
        let dep' := fun k => dep.getD k false
        let shape := packShape rn d w dep' g.a
        let reads := (points (fillBox rn d dep' b)).map fun o => HEv.read dv.loc dv.tag (dv.idx dv.q (vecOf o))
        match g.store with
        | some (sid, _) =>
          reads ++ ([.fill sid shape b] ++ htrace rn ro slot body (packG d w g) (packD rn d w dep' sid b) b)
        | none =>
          [.alloc (g.depth + 1) (prod shape)] ++ (reads ++ ([.fill (g.depth + 1) shape b] ++
            (htrace rn ro slot body { packG d w g with depth := g.depth + 1 } (packD rn d w dep' (g.depth + 1) b) b ++
              [.free (g.depth + 1)])))
      else [.fault]
    else htrace rn ro slot body g dv b
  | .stage s body, g0, dv, b =>
    let g := ownStage s g0
    if s = slot ∧ g.ready = false then
      match g.store with
      | some (sid, _) => [.fill sid (stageShape g b) b] ++ htrace rn ro slot body (stageG g) (stageD g sid b) b
      | none =>
        [.alloc (g.depth + 1) (prod (stageShape g b)), .fill (g.depth + 1) (stageShape g b) b] ++
          (htrace rn ro slot body { stageG g with depth := g.depth + 1 } (stageD g (g.depth + 1) b) b ++
            [.free (g.depth + 1)])
    else htrace rn ro slot body g dv b
  | .alloc s body, g, dv, b =>
    if s = slot ∧ allocSize rn slot g body ≠ 0 then
      [.alloc (g.depth + 1) (allocSize rn slot g body)] ++
        (htrace rn ro slot body
            { g with store := some (g.depth + 1, allocSize rn slot g body), depth := g.depth + 1 } dv b ++
          [.free (g.depth + 1)])
    else htrace rn ro slot body g dv b
  | .switch d c small big, g, dv, b =>
    match b[d]?, g.a[d]? with
    | some (_, n), some x =>
      if n < c then htrace rn ro slot small { g with a := g.a.set d (asmall c x) } dv b
      else htrace rn ro slot big g dv b
    | _, _ => []

/-- The checks that make a run safe on the heap, on the static part of the state alone. -/
def hcheck (rn ro slot : Nat) : Tree → HG → Bool
  | .leaf vdim, g =>
    g.ready &&
      (!full g vdim ||
        (List.range ro).all fun e =>
          match g.a[e]? with
          | some x => !x.dyn || !g.reads e || decide (x.cap ≤ g.unit e)
          | none => true)
  | .split d ts mode body, g =>
    match g.a[d]? with
    | some x =>
      decide (0 < ts) && decide (ts % g.unit d = 0) &&
        (atiles ts mode x).all fun y => hcheck rn ro slot body { g with a := g.a.set d y }
    | none => false
  | .reduce _ _ body, g => hcheck rn ro slot body g
  | .pack d w s dep body, g0 =>
    let g := ownPack s g0
    if s = slot then
      g.ready && decide (0 < w) && decide (d < rn) && g.reads d &&
        ((List.range rn).all fun k => dep.getD k false == g.reads k) &&
        match g.store with
        | some (_, cap) =>
          decide (prod (packShape rn d w (fun k => dep.getD k false) g.a) ≤ cap) && hcheck rn ro slot body (packG d w g)
        | none => hcheck rn ro slot body { packG d w g with depth := g.depth + 1 }
    else hcheck rn ro slot body g
  | .stage s body, g0 =>
    let g := ownStage s g0
    if s = slot ∧ g.ready = false then
      match g.store with
      | some (_, cap) => decide (stageBound g (capOf1 g.a) ≤ cap) && hcheck rn ro slot body (stageG g)
      | none => hcheck rn ro slot body { stageG g with depth := g.depth + 1 }
    else hcheck rn ro slot body g
  | .alloc s body, g =>
    if s = slot ∧ allocSize rn slot g body ≠ 0 then
      hcheck rn ro slot body { g with store := some (g.depth + 1, allocSize rn slot g body), depth := g.depth + 1 }
    else hcheck rn ro slot body g
  | .switch d c small big, g =>
    match g.a[d]? with
    | some x =>
      if x.dyn then hcheck rn ro slot small { g with a := g.a.set d (asmall c x) } && hcheck rn ro slot big g
      else if x.cap < c then hcheck rn ro slot small { g with a := g.a.set d (asmall c x) } else hcheck rn ro slot big g
    | none => false

end Sched
