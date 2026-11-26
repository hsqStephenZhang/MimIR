/-!
# Schedule trees and their value semantics

A model of the `SchedTree` nodes that cut the iteration space (`Leaf`, `Split`, `Switch`) and of `run_levels_at` on them.
A run is the list of boxes its leaves compute, in execution order.
The theorems say that these boxes hold every point of the root box once, and that a reduction over them therefore computes the specification.
-/

namespace Sched

/-- An axis of a box: where it starts and how long it is. -/
abbrev Tile := Nat × Nat
abbrev Box := List Tile
abbrev Point := List Nat

/-- The points of a box in lexicographic order, the order in which a `Leaf` visits them. -/
def points : Box → List Point
  | [] => [[]]
  | (lo, n) :: r => (List.range' lo n).flatMap fun i => (points r).map (i :: ·)

/-- `grid` and `threads` bind the tiles to GPU work-groups and work-items, and `piped` is a serial loop that stages a tile ahead. -/
inductive Mode where
  | fixed | loop | peel | grid | threads | piped
  deriving Repr, DecidableEq

inductive Tree where
  | leaf   (vdim : Nat)
  | split  (d ts : Nat) (mode : Mode) (body : Tree)
  | reduce (d w : Nat) (body : Tree)
  | pack   (d w slot : Nat) (dep : List Bool) (body : Tree)
  | stage  (slot : Nat) (body : Tree)
  | alloc  (slot : Nat) (body : Tree)
  | switch (d c : Nat) (small big : Tree)
  deriving Repr

/-- The tiles of size `ts` that cut `[lo, lo + n)`, the last of which may be shorter. -/
def tiles (ts lo n : Nat) : List Tile :=
  if _h : 0 < ts ∧ 0 < n then (lo, min n ts) :: tiles ts (lo + min n ts) (n - min n ts) else []
termination_by n
decreasing_by omega

def run : Tree → Box → List Box
  | .leaf _, b => [b]
  | .split d ts _ body, b =>
    match b[d]? with
    | some (lo, n) => (tiles ts lo n).flatMap fun tl => run body (b.set d tl)
    | none => []
  | .reduce _ _ body, b => run body b
  | .pack _ _ _ _ body, b => run body b
  | .stage _ body, b => run body b
  | .alloc _ body, b => run body b
  | .switch d c small big, b =>
    match b[d]? with
    | some (_, n) => if n < c then run small b else run big b
    | none => []

/-- The checks an instruction runs on its own fields, for `rn` loops of which the first `ro` are output loops. -/
def Tree.wf (rn ro : Nat) : Tree → Bool
  | .leaf vdim => vdim ≤ rn
  | .split d ts _ body => d < rn && 0 < ts && body.wf rn ro
  | .reduce d w body => ro ≤ d && d < rn && 1 < w && body.wf rn ro
  | .pack d w _ _ body => d < rn && 0 < w && body.wf rn ro
  | .stage _ body => body.wf rn ro
  | .alloc _ body => body.wf rn ro
  | .switch d c small big => d < rn && 1 < c && small.wf rn ro && big.wf rn ro

theorem tiles_cover {ts : Nat} (hts : 0 < ts) (lo n : Nat) :
    (tiles ts lo n).flatMap (fun tl => List.range' tl.1 tl.2) = List.range' lo n := by
  induction n using Nat.strongRecOn generalizing lo with
  | _ n ih =>
    unfold tiles
    split
    · rename_i h
      rw [List.flatMap_cons, ih (n - min n ts) (by omega)]
      have : n = min n ts + (n - min n ts) := by omega
      conv => rhs; rw [this]
      exact List.range'_append_1
    · rename_i h
      have : n = 0 := by omega
      simp [this]

/-- A tile is a whole one or the remainder. -/
theorem mem_tiles {ts lo n : Nat} {tl : Tile} (h : tl ∈ tiles ts lo n) :
    0 < ts ∧ (tl.2 = ts ∧ ts ≤ n ∨ tl.2 = n % ts ∧ n % ts ≠ 0) := by
  induction n using Nat.strongRecOn generalizing lo with
  | _ n ih =>
    unfold tiles at h
    split at h
    · rename_i hn
      rcases List.mem_cons.1 h with rfl | h
      · refine ⟨hn.1, ?_⟩
        by_cases hle : ts ≤ n
        · exact .inl ⟨Nat.min_eq_right hle, hle⟩
        · have : n % ts = n := Nat.mod_eq_of_lt (by omega)
          exact .inr ⟨by simp only [this]; omega, by omega⟩
      · have := ih (n - min n ts) (by omega) h
        refine ⟨hn.1, ?_⟩
        by_cases hle : ts ≤ n
        · have hm : (n - min n ts) % ts = n % ts := by
            rw [Nat.min_eq_right hle]
            exact (Nat.mod_eq_sub_mod hle).symm
          rw [hm] at this
          rcases this.2 with ⟨h1, h2⟩ | h2
          · exact .inl ⟨h1, hle⟩
          · exact .inr h2
        · have : n - min n ts = 0 := by omega
          rw [this] at h
          unfold tiles at h
          simp at h
    · simp at h

theorem flatMap_append_perm {α β} (l : List α) (f g : α → List β) :
    l.flatMap (fun a => f a ++ g a) |>.Perm (l.flatMap f ++ l.flatMap g) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.flatMap_cons]
    refine ((List.Perm.append_left _ ih).trans ?_)
    simp only [List.append_assoc]
    refine List.Perm.append_left _ ?_
    simp only [← List.append_assoc]
    exact List.Perm.append_right _ List.perm_append_comm

theorem flatMap_comm_perm {α β γ} (l₁ : List α) (l₂ : List β) (f : α → β → List γ) :
    (l₁.flatMap fun a => l₂.flatMap fun b => f a b).Perm (l₂.flatMap fun b => l₁.flatMap fun a => f a b) := by
  induction l₁ with
  | nil => simp
  | cons a l ih =>
    simp only [List.flatMap_cons]
    exact (List.Perm.append_left _ ih).trans (flatMap_append_perm l₂ _ _).symm

theorem flatMap_perm_congr {α β} (l : List α) {f g : α → List β} (h : ∀ a ∈ l, (f a).Perm (g a)) :
    (l.flatMap f).Perm (l.flatMap g) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.flatMap_cons]
    exact (h a (by simp)).append (ih fun b hb => h b (by simp [hb]))

/-- Cutting axis `d` into tiles that cover it visits the points of the box once. -/
theorem points_set_perm (ts : List Tile) :
    ∀ (b : Box) (d lo n : Nat), b[d]? = some (lo, n) →
      ts.flatMap (fun tl => List.range' tl.1 tl.2) = List.range' lo n →
      (ts.flatMap fun tl => points (b.set d tl)).Perm (points b)
  | [], _, _, _, h, _ => by simp at h
  | hd :: r, 0, lo, n, h, hc => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp only [List.set_cons_zero, points]
    rw [← hc, List.flatMap_assoc]
  | (l0, n0) :: r, d + 1, lo, n, h, hc => by
    simp only [List.getElem?_cons_succ] at h
    have ih := points_set_perm ts r d lo n h hc
    simp only [List.set_cons_succ, points]
    refine (flatMap_comm_perm _ _ _).trans ?_
    refine flatMap_perm_congr _ fun i _ => ?_
    rw [← List.map_flatMap]
    exact ih.map _

theorem mem_tiles_range {ts lo n : Nat} {tl : Tile} (h : tl ∈ tiles ts lo n) :
    lo ≤ tl.1 ∧ tl.1 + tl.2 ≤ lo + n ∧ 0 < tl.2 := by
  induction n using Nat.strongRecOn generalizing lo with
  | _ n ih =>
    unfold tiles at h
    split at h
    · rename_i hn
      rcases List.mem_cons.1 h with rfl | h
      · simp only; omega
      · have := ih (n - min n ts) (by omega) h
        omega
    · simp at h

theorem tiles_single {ts lo n : Nat} (hts : 0 < ts) (hn : 0 < n) (h : n ≤ ts) : tiles ts lo n = [(lo, n)] := by
  unfold tiles
  rw [dite_eq_left ⟨hts, hn⟩, Nat.min_eq_left h, Nat.sub_self]
  unfold tiles
  simp

/-- A fold over the tiles of an interval is one step on the interval when two adjacent steps make one. -/
theorem tiles_foldl {S : Type} (G : Tile → S → S) {ts : Nat} (hts : 0 < ts) (lo n : Nat)
    (hseq : ∀ l m k s, lo ≤ l → l + m + k ≤ lo + n → 0 < m → 0 < k → G (l + m, k) (G (l, m) s) = G (l, m + k) s) :
    ∀ n' l s, lo ≤ l → l + n' ≤ lo + n → 0 < n' → (tiles ts l n').foldl (fun s tl => G tl s) s = G (l, n') s := by
  intro n'
  induction n' using Nat.strongRecOn with
  | _ n' ih =>
    intro l s hl hle hn
    unfold tiles
    rw [dite_eq_left ⟨hts, hn⟩, List.foldl_cons]
    by_cases hm : n' ≤ ts
    · rw [Nat.min_eq_left hm, Nat.sub_self]
      unfold tiles
      simp
    · have hmin : min n' ts = ts := by omega
      rw [hmin, ih (n' - ts) (by omega) (l + ts) _ (by omega) (by omega) (by omega),
        hseq l ts (n' - ts) s hl (by omega) hts (by omega)]
      congr 2
      omega

theorem mem_points_cons {lo n : Nat} {r : Box} {c : Point} :
    c ∈ points ((lo, n) :: r) ↔ ∃ i p, c = i :: p ∧ lo ≤ i ∧ i < lo + n ∧ p ∈ points r := by
  simp only [points, List.mem_flatMap, List.mem_range'_1, List.mem_map]
  constructor
  · rintro ⟨i, ⟨h1, h2⟩, p, hp, rfl⟩
    exact ⟨i, p, rfl, h1, h2, hp⟩
  · rintro ⟨i, p, rfl, h1, h2, hp⟩
    exact ⟨i, ⟨h1, h2⟩, p, hp, rfl⟩

theorem points_eq_nil_iff : ∀ {b : Box}, points b = [] ↔ ∃ tl ∈ b, tl.2 = 0
  | [] => by simp [points]
  | (lo, n) :: r => by
    have ih := @points_eq_nil_iff r
    constructor
    · intro h
      by_cases hn : n = 0
      · exact ⟨(lo, n), by simp, hn⟩
      · by_cases hr : points r = []
        · obtain ⟨tl, htl, h0⟩ := ih.1 hr
          exact ⟨tl, by simp [htl], h0⟩
        · obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hr
          have : lo :: p ∈ points ((lo, n) :: r) := mem_points_cons.2 ⟨lo, p, rfl, Nat.le_refl _, by omega, hp⟩
          rw [h] at this
          simp at this
    · rintro ⟨tl, htl, h0⟩
      rcases List.mem_cons.1 htl with rfl | htl
      · simp only at h0
        simp [points, h0]
      · have := ih.2 ⟨tl, htl, h0⟩
        simp [points, this]

theorem length_of_mem_run {t : Tree} : ∀ {b leaf : Box}, leaf ∈ run t b → leaf.length = b.length := by
  induction t with
  | leaf _ => intro b leaf h; simp [run] at h; rw [h]
  | split d ts m body ih =>
    intro b leaf h
    simp only [run] at h
    split at h
    · obtain ⟨tl, _, h⟩ := List.mem_flatMap.1 h
      simpa using ih h
    · simp at h
  | reduce _ _ body ih | pack _ _ _ _ body ih | stage _ body ih | alloc _ body ih => intro b leaf h; exact ih h
  | switch d c s g ihs ihg =>
    intro b leaf h
    simp only [run] at h
    split at h
    · split at h
      · exact ihs h
      · exact ihg h
    · simp at h

/-- The leaves of a well-formed tree hold every point of the root box once. -/
theorem run_perm {rn ro : Nat} : ∀ (t : Tree) (b : Box), t.wf rn ro = true → b.length = rn →
    ((run t b).flatMap points).Perm (points b)
  | .leaf _, b, _, _ => by simp [run]
  | .split d ts m body, b, h, hb => by
    simp only [Tree.wf, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨hd, hts⟩, hbody⟩ := h
    simp only [run]
    have hd' : d < b.length := hb ▸ hd
    rw [List.getElem?_eq_getElem hd']
    rcases hx : b[d] with ⟨lo, n⟩
    simp only
    rw [List.flatMap_assoc]
    refine (flatMap_perm_congr _ fun tl _ => ?_).trans
      (points_set_perm (tiles ts lo n) b d lo n (by rw [List.getElem?_eq_getElem hd', hx]) (tiles_cover hts lo n))
    exact run_perm body (b.set d tl) hbody (by simpa using hb)
  | .reduce _ _ body, b, h, hb => by
    simp only [Tree.wf, Bool.and_eq_true] at h
    exact run_perm body b h.2 hb
  | .pack _ _ _ _ body, b, h, hb => by
    simp only [Tree.wf, Bool.and_eq_true] at h
    exact run_perm body b h.2 hb
  | .stage _ body, b, h, hb => run_perm body b h hb
  | .alloc _ body, b, h, hb => run_perm body b h hb
  | .switch d c s g, b, h, hb => by
    simp only [Tree.wf, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨hd, _⟩, hs⟩, hg⟩ := h
    simp only [run]
    rw [List.getElem?_eq_getElem (hb ▸ hd)]
    rcases b[d] with ⟨lo, n⟩
    simp only
    split
    · exact run_perm s b hs hb
    · exact run_perm g b hg hb

end Sched
