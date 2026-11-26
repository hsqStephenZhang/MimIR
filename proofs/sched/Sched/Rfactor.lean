import Sched.Exact

/-!
# `Reduce`: partial results

A `Leaf` under a `Reduce` adds every `w`-th element of its reduction into one of `w` partial results and merges them at the end.
With a merge that makes the accumulators a commutative monoid, this is the left fold.
-/

namespace Sched

variable {α β : Type}

/-- What a combiner with a merge promises. -/
structure Merge (f : α → β → α) (init : α) (merge : α → α → α) : Prop where
  assoc : ∀ a b c, merge (merge a b) c = merge a (merge b c)
  comm  : ∀ a b, merge a b = merge b a
  ident : ∀ a, merge a init = a
  split : ∀ a x, f a x = merge a (f init x)

variable {f : α → β → α} {init : α} {merge : α → α → α}

theorem Merge.foldl (h : Merge f init merge) (l : List β) (a : α) : l.foldl f a = merge a (l.foldl f init) := by
  induction l generalizing a with
  | nil => exact (h.ident a).symm
  | cons x l ih =>
    simp only [List.foldl_cons]
    rw [ih (f a x), ih (f init x), h.split a x, h.assoc]

/-- A merge makes the order of the additions irrelevant, which is what `Op.exec_spec` asks of a tree that cuts several reduction loops. -/
theorem Merge.comm_f (h : Merge f init merge) (a : α) (x y : β) : f (f a x) y = f (f a y) x := by
  rw [h.split (f a x) y, h.split a x, h.split (f a y) x, h.split a y, h.assoc, h.assoc, h.comm (f init x)]

theorem Merge.parts (h : Merge f init merge) (parts : List (List β)) (a : α) :
    parts.foldl (fun acc p => merge acc (p.foldl f init)) a = parts.flatten.foldl f a := by
  induction parts generalizing a with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.foldl_cons, List.flatten_cons, List.foldl_append]
    rw [ih, ← h.foldl p a]

theorem flatMap_single {w k : Nat} (x : β) (hk : k < w) :
    ((List.range w).flatMap fun i => if k = i then [x] else []) = [x] := by
  induction w with
  | zero => omega
  | succ w ih =>
    rw [List.range_succ, List.flatMap_append]
    by_cases hkw : k = w
    · subst hkw
      have : ((List.range k).flatMap fun i => if k = i then [x] else []) = [] := by
        refine List.flatMap_eq_nil_iff.2 fun i hi => ?_
        have := List.mem_range.1 hi
        have hne : ¬k = i := by omega
        simp [hne]
      simp [this]
    · rw [ih (by omega)]
      simp [hkw]

/-- The lanes hold every element once. -/
theorem lanes_perm (w : Nat) (key : β → Nat) (l : List β) (hk : ∀ x ∈ l, key x < w) :
    ((List.range w).flatMap fun i => l.filter fun x => key x = i).Perm l := by
  induction l with
  | nil => simp
  | cons x l ih =>
    have he : (fun i => (x :: l).filter fun y => key y = i) =
        fun i => (if key x = i then [x] else []) ++ l.filter fun y => key y = i := by
      funext i
      by_cases hx : key x = i <;> simp [hx]
    rw [he]
    refine (flatMap_append_perm _ _ _).trans ?_
    rw [flatMap_single x (hk x (by simp))]
    exact (ih fun y hy => hk y (by simp [hy])).cons x

/-- Folding `w` lanes from `init` and merging them into the accumulator is the left fold. -/
theorem Merge.rfactor (h : Merge f init merge) (w : Nat) (key : β → Nat) (l : List β) (hk : ∀ x ∈ l, key x < w)
    (a : α) :
    ((List.range w).map fun i => l.filter fun x => key x = i).foldl (fun acc p => merge acc (p.foldl f init)) a =
      l.foldl f a := by
  rw [h.parts, ← List.flatMap_def]
  exact (lanes_perm w key l hk).foldl_eq' (fun x _ y _ z => h.comm_f z x y) a

/-- The reduction of a cell in `w` lanes along a key of its points, e.g. the coordinate of the loop its `Leaf` vectorizes modulo `w`. -/
def Op.redLanes (op : Op β α) (merge : α → α → α) (w : Nat) (key : Point → Nat) (r : Box) (c : Point) (a : α) : α :=
  ((List.range w).map fun i => (points r).filter fun p => key p = i).foldl
    (fun acc ps => merge acc (ps.foldl (fun a p => op.f a (op.inp (c ++ p))) op.init)) a

theorem Op.redLanes_eq (op : Op β α) {merge : α → α → α} (h : Merge op.f op.init merge) (w : Nat) (key : Point → Nat)
    (r : Box) (c : Point) (a : α) (hk : ∀ p ∈ points r, key p < w) : op.redLanes merge w key r c a = op.red r c a := by
  have h' : Merge (fun a p => op.f a (op.inp (c ++ p))) op.init merge :=
    ⟨h.assoc, h.comm, h.ident, fun a p => h.split a _⟩
  exact h'.rfactor w key (points r) hk a

end Sched
