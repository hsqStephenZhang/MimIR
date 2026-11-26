import Sched.Tree

/-!
# Ghost state

The static part of an axis register of `run_levels_at`: the bound `cap` on its extent, whether the extent is only known at run time, whether a `Split` cut the axis, and the partial results a `Reduce` asked for.
Partial evaluation advances this state along the tree without the run-time extents and evaluates the checks on it.
The theorem says that the state it reaches at a leaf describes every box that leaf computes at run time.
-/

namespace Sched

structure AAxis where
  cap  : Nat
  dyn  : Bool
  part : Bool := false
  red  : Nat := 0
  deriving Repr, DecidableEq

abbrev AState := List AAxis

/-- A static extent is its `cap`; a run-time extent stays within it. -/
def AAxis.Holds (x : AAxis) (tl : Tile) : Prop :=
  tl.2 ≤ x.cap ∧ (x.dyn = false → tl.2 = x.cap)

def Holds (a : AState) (b : Box) : Prop :=
  a.length = b.length ∧ ∀ i (ha : i < a.length) (hb : i < b.length), a[i].Holds b[i]

/-- The states `split_axis` runs its body in: whole tiles and a remainder when it peels them, one counted tile otherwise. -/
def atiles (ts : Nat) (mode : Mode) (x : AAxis) : List AAxis :=
  let part := x.part || x.dyn || decide (ts < x.cap)
  if x.dyn then
    if mode = .peel then [⟨ts, false, part, x.red⟩, ⟨min x.cap ts, true, part, x.red⟩]
    else [⟨min x.cap ts, true, part, x.red⟩]
  else if mode = .loop ∧ x.cap / ts ≠ 0 ∧ x.cap % ts ≠ 0 then [⟨min x.cap ts, true, part, x.red⟩]
  else (if x.cap / ts ≠ 0 then [⟨ts, false, part, x.red⟩] else []) ++
    (if x.cap % ts ≠ 0 then [⟨x.cap % ts, false, part, x.red⟩] else [])

/-- The small branch of a `Switch` may size its scratch by `c - 1`. -/
def asmall (c : Nat) (x : AAxis) : AAxis := { x with cap := min x.cap (c - 1) }

/-- The ghost states at the leaves. -/
def aleaves : Tree → AState → List AState
  | .leaf _, a => [a]
  | .split d ts mode body, a =>
    match a[d]? with
    | some x => (atiles ts mode x).flatMap fun y => aleaves body (a.set d y)
    | none => []
  | .reduce d w body, a =>
    match a[d]? with
    | some x => aleaves body (a.set d { x with red := w })
    | none => []
  | .pack _ _ _ _ body, a => aleaves body a
  | .stage _ body, a => aleaves body a
  | .alloc _ body, a => aleaves body a
  | .switch d c small big, a =>
    match a[d]? with
    | some x =>
      if x.dyn then aleaves small (a.set d (asmall c x)) ++ aleaves big a
      else if x.cap < c then aleaves small (a.set d (asmall c x)) else aleaves big a
    | none => []

/-- No reduction loop after the first one is cut, so a cell's additions keep their order. -/
def ordered (ro : Nat) (a : AState) : Bool := (a.drop (ro + 1)).all fun x => !x.part

/-- `levels_unbounded`: the `cap` of a run-time extent that no `Split` bounds yet. -/
def unbounded : Nat := 1099511627776

def dynAt (a : AState) (d : Nat) : Bool :=
  match a[d]? with
  | some x => x.dyn
  | none => false

/-- The checks of a `Leaf` for a combiner with `lanes` partial results, where `0` means that it has no merge. -/
def leafOk (lanes ro vdim : Nat) (a : AState) : Bool :=
  (0 < lanes || ordered ro a) &&
    ((List.range a.length).all fun d =>
      match a[d]? with
      | some x => if d = vdim then x.red ≤ 1 || 0 < lanes else x.red = 0
      | none => true) &&
    (!(dynAt a vdim || dynAt a (vdim + 1)) || (a.take ro).all fun x => x.cap < unbounded)

/-- A `Pack` sizes its panel by the `cap` of its axis and of the loops its slot reads. -/
def packOk (d : Nat) (dep : List Bool) (a : AState) : Bool :=
  (List.range a.length).all fun e =>
    match a[e]? with
    | some x => !(dep.getD e false || e = d) || x.cap < unbounded
    | none => true

/-- The checks that depend on the ghost state. -/
def scheck (lanes ro : Nat) : Tree → AState → Bool
  | .leaf vdim, a => leafOk lanes ro vdim a
  | .split d ts mode body, a =>
    match a[d]? with
    | some x => !(mode ≠ .loop && mode ≠ .peel && x.dyn) && (atiles ts mode x).all fun y => scheck lanes ro body (a.set d y)
    | none => false
  | .reduce d w body, a =>
    match a[d]? with
    | some x => scheck lanes ro body (a.set d { x with red := w })
    | none => false
  | .pack d _ _ dep body, a => packOk d dep a && scheck lanes ro body a
  | .stage _ body, a => scheck lanes ro body a
  | .alloc _ body, a => scheck lanes ro body a
  | .switch d c small big, a =>
    match a[d]? with
    | some x =>
      if x.dyn then scheck lanes ro small (a.set d (asmall c x)) && scheck lanes ro big a
      else if x.cap < c then scheck lanes ro small (a.set d (asmall c x)) else scheck lanes ro big a
    | none => false

/-- The checks of the interpreter, evaluated on ghost state alone. -/
def check (lanes ro : Nat) (t : Tree) (a : AState) : Bool := t.wf a.length ro && scheck lanes ro t a

theorem check_wf {lanes ro : Nat} {t : Tree} {a : AState} (h : check lanes ro t a = true) : t.wf a.length ro = true := by
  simp only [check, Bool.and_eq_true] at h
  exact h.1

theorem Holds.set {a : AState} {b : Box} (h : Holds a b) (d : Nat) {y : AAxis} {tl : Tile} (hy : y.Holds tl) :
    Holds (a.set d y) (b.set d tl) := by
  refine ⟨by simpa using h.1, fun i ha hb => ?_⟩
  simp only [List.getElem_set]
  split
  · exact hy
  · exact h.2 i (by simpa using ha) (by simpa using hb)

theorem atiles_sound {ts lo n : Nat} {mode : Mode} {x : AAxis} {tl : Tile}
    (hx : x.Holds (lo, n)) (htl : tl ∈ tiles ts lo n) :
    ∃ y ∈ atiles ts mode x, y.Holds tl := by
  obtain ⟨hts, hsz⟩ := mem_tiles htl
  obtain ⟨hcap, hst⟩ := hx
  simp only at hcap hst
  have hmod := Nat.mod_lt n hts
  have hmle := Nat.mod_le n ts
  have hbound : tl.2 ≤ min x.cap ts := by rcases hsz with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
  simp only [atiles]
  generalize (x.part || x.dyn || decide (ts < x.cap)) = part
  by_cases hdyn : x.dyn = true
  · simp only [hdyn, ite_true]
    by_cases hp : mode = .peel
    · simp only [hp, ite_true]
      rcases hsz with ⟨h, _⟩ | ⟨h, _⟩
      · exact ⟨⟨ts, false, part, x.red⟩, by simp, by simp [AAxis.Holds, h]⟩
      · exact ⟨⟨min x.cap ts, true, part, x.red⟩, by simp, ⟨hbound, by simp⟩⟩
    · simp only [hp, ite_false]
      exact ⟨⟨min x.cap ts, true, part, x.red⟩, by simp, ⟨hbound, by simp⟩⟩
  · have hf : x.dyn = false := by simpa using hdyn
    have hn := hst hf
    simp only [hf, Bool.false_eq_true, ite_false]
    rw [← hn]
    by_cases hl : mode = .loop ∧ n / ts ≠ 0 ∧ n % ts ≠ 0
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hl)]
      exact ⟨⟨min n ts, true, part, x.red⟩, by simp, ⟨hn ▸ hbound, by simp⟩⟩
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hl)]
      rcases hsz with ⟨h, hle⟩ | ⟨h, hne⟩
      · have : n / ts ≠ 0 := Nat.ne_of_gt (Nat.div_pos hle hts)
        exact ⟨⟨ts, false, part, x.red⟩,
          List.mem_append_left _ (by rw [ite_eq_left_of_eq_true _ _ (eq_true this)]; simp), by simp [AAxis.Holds, h]⟩
      · exact ⟨⟨n % ts, false, part, x.red⟩,
          List.mem_append_right _ (by rw [ite_eq_left_of_eq_true _ _ (eq_true hne)]; simp), by simp [AAxis.Holds, h]⟩

/-- Whatever the run-time extents are, the box a leaf computes satisfies one of the ghost states the checks saw there. -/
theorem ghost_sound {lanes ro : Nat} : ∀ (t : Tree) (a : AState) (b : Box), scheck lanes ro t a = true → Holds a b →
    ∀ leaf ∈ run t b, ∃ a' ∈ aleaves t a, Holds a' leaf
  | .leaf _, a, b, _, h, leaf, hl => by
    simp only [run, List.mem_singleton] at hl
    exact ⟨a, by simp [aleaves], hl ▸ h⟩
  | .split d ts mode body, a, b, hc, h, leaf, hl => by
    simp only [scheck] at hc
    simp only [run] at hl
    simp only [aleaves]
    split at hc
    · rename_i x hx
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hdb : d < b.length := h.1 ▸ hda
      rw [List.getElem?_eq_getElem hdb] at hl
      have hxb : x.Holds b[d] := by
        have := h.2 d hda hdb
        rwa [(List.getElem?_eq_some_iff.1 hx).2] at this
      rcases hbd : b[d] with ⟨lo, n⟩
      rw [hbd] at hl hxb
      simp only [Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true] at hc
      obtain ⟨_, hall⟩ := hc
      obtain ⟨tl, htl, hl⟩ := List.mem_flatMap.1 hl
      obtain ⟨y, hy, hytl⟩ := atiles_sound (mode := mode) hxb htl
      obtain ⟨a', ha', hh⟩ := ghost_sound body (a.set d y) (b.set d tl) (hall y hy) (h.set d hytl) leaf hl
      exact ⟨a', List.mem_flatMap.2 ⟨y, hy, ha'⟩, hh⟩
    · simp at hc
  | .reduce d w body, a, b, hc, h, leaf, hl => by
    simp only [scheck] at hc
    simp only [aleaves]
    split at hc
    · rename_i x hx
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hdb : d < b.length := h.1 ▸ hda
      have hxb : x.Holds b[d] := by
        have := h.2 d hda hdb
        rwa [(List.getElem?_eq_some_iff.1 hx).2] at this
      have hh : Holds (a.set d { x with red := w }) b := by
        have := h.set d (y := { x with red := w }) (tl := b[d]) hxb
        rwa [List.set_getElem_self hdb] at this
      exact ghost_sound body _ b hc hh leaf hl
    · simp at hc
  | .pack _ _ _ _ body, a, b, hc, h, leaf, hl => by
    simp only [scheck, Bool.and_eq_true] at hc
    exact ghost_sound body a b hc.2 h leaf hl
  | .stage _ body, a, b, hc, h, leaf, hl => ghost_sound body a b hc h leaf hl
  | .alloc _ body, a, b, hc, h, leaf, hl => ghost_sound body a b hc h leaf hl
  | .switch d c small big, a, b, hc, h, leaf, hl => by
    simp only [scheck] at hc
    simp only [run] at hl
    simp only [aleaves]
    split at hc
    · rename_i x hx
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hdb : d < b.length := h.1 ▸ hda
      rw [List.getElem?_eq_getElem hdb] at hl
      have hxb : x.Holds b[d] := by
        have := h.2 d hda hdb
        rwa [(List.getElem?_eq_some_iff.1 hx).2] at this
      have hself : b.set d b[d] = b := List.set_getElem_self hdb
      rcases hbd : b[d] with ⟨lo, n⟩
      rw [hbd] at hl hxb hself
      obtain ⟨hcap, hst⟩ := hxb
      simp only at hcap hst hl
      have hsmall : n < c → Holds (a.set d (asmall c x)) b := fun hn => by
        have : (asmall c x).Holds (lo, n) := by
          refine ⟨by simp only [asmall]; omega, fun hd => ?_⟩
          have := hst hd
          simp only [asmall]
          omega
        simpa [hself] using h.set d this
      by_cases hdyn : x.dyn = true
      · simp only [hdyn, ite_true, Bool.and_eq_true] at hc ⊢
        split at hl
        · rename_i hn
          obtain ⟨a', ha', hh⟩ := ghost_sound small _ b hc.1 (hsmall hn) leaf hl
          exact ⟨a', List.mem_append_left _ ha', hh⟩
        · obtain ⟨a', ha', hh⟩ := ghost_sound big a b hc.2 h leaf hl
          exact ⟨a', List.mem_append_right _ ha', hh⟩
      · have hn := hst (by simpa using hdyn)
        simp only [hdyn] at hc ⊢
        subst hn
        split at hl
        · rename_i hn
          simp only [hn, ite_true] at hc ⊢
          exact ghost_sound small _ b hc (hsmall hn) leaf hl
        · rename_i hn
          simp only [hn, ite_false] at hc ⊢
          exact ghost_sound big a b hc h leaf hl
    · simp at hc

/-- A buffer that a leaf sizes by `cap` holds its tile: a bound on the ghost state bounds every run-time extent. -/
theorem leaf_bounded {lanes ro : Nat} {t : Tree} {a : AState} {b : Box} (hc : scheck lanes ro t a = true) (h : Holds a b) {k : Nat}
    (hk : ∀ a' ∈ aleaves t a, ∀ x ∈ a', x.cap ≤ k) : ∀ leaf ∈ run t b, ∀ tl ∈ leaf, tl.2 ≤ k := by
  intro leaf hl tl htl
  obtain ⟨a', ha', hh⟩ := ghost_sound t a b hc h leaf hl
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem htl
  have hia : i < a'.length := hh.1 ▸ hi
  exact Nat.le_trans (hh.2 i hia hi).1 (hk a' ha' _ (List.getElem_mem hia))

/-- GEMM over `(m, n, k)` with a run-time `m`: blocks of 96 with register tiles of 6 peeled off, and `k` cut by 256. -/
def gemmTree : Tree :=
  .split 2 256 .fixed (.split 0 96 .loop (.split 0 6 .peel (.split 1 8 .fixed (.leaf 0))))

example : check 0 2 gemmTree [⟨4096, true, false, 0⟩, ⟨512, false, false, 0⟩, ⟨1000, false, false, 0⟩] = true := by decide

example : (aleaves gemmTree [⟨4096, true, false, 0⟩, ⟨512, false, false, 0⟩, ⟨1000, false, false, 0⟩]).map (·.map (·.cap)) =
    [[6, 8, 256], [6, 8, 256], [6, 8, 232], [6, 8, 232]] := by decide

/-- A `Fixed` split of a run-time axis is rejected. -/
example : check 0 1 (.split 0 6 .fixed (.leaf 0)) [⟨4096, true, false, 0⟩] = false := by decide

/-- A combiner without a merge may not have its second reduction loop cut. -/
example : check 0 1 (.split 2 4 .fixed (.leaf 0)) [⟨8, false, false, 0⟩, ⟨8, false, false, 0⟩, ⟨8, false, false, 0⟩] = false := by
  decide

end Sched
