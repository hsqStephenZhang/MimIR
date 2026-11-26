import Sched.Machine

/-!
# Splits that bind their tiles to GPU work-groups and work-items

`Grid` and `Threads` cut a loop into the same tiles as a serial `Split`, but the tiles run in no fixed order.
For an output loop the order does not matter: the tiles write disjoint cells.
`Strided` hands a work-item every `n`-th element, which also visits every element once.
-/

namespace Sched

variable {β α : Type}

/-- Leaves on boxes without a common cell commute. -/
theorem Op.leaf_comm (op : Op β α) (ro : Nat) (rb b₁ b₂ : Box) (dst : Point → α)
    (h : ∀ c, ¬(c ∈ points (b₁.take ro) ∧ c ∈ points (b₂.take ro))) :
    op.leaf ro rb b₁ (op.leaf ro rb b₂ dst) = op.leaf ro rb b₂ (op.leaf ro rb b₁ dst) := by
  funext c
  simp only [Op.leaf]
  by_cases h1 : c ∈ points (b₁.take ro)
  · have h2 : c ∉ points (b₂.take ro) := fun h2 => h c ⟨h1, h2⟩
    simp only [h1, h2, ↓reduceIte]
  · by_cases h2 : c ∈ points (b₂.take ro) <;> simp only [h1, h2, ↓reduceIte]

/-- The tiles of a loop come in order and do not overlap. -/
theorem tiles_pairwise (ts : Nat) : ∀ n lo, (tiles ts lo n).Pairwise fun t₁ t₂ => t₁.1 + t₁.2 ≤ t₂.1 := by
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
    intro lo
    unfold tiles
    split
    · rename_i h
      refine List.pairwise_cons.2 ⟨fun t ht => ?_, ih (n - min n ts) (by omega) _⟩
      exact (mem_tiles_range ht).1
    · exact List.Pairwise.nil

theorem tiles_disjoint {ts lo n : Nat} {t₁ t₂ : Tile} (h₁ : t₁ ∈ tiles ts lo n) (h₂ : t₂ ∈ tiles ts lo n)
    (hne : t₁ ≠ t₂) : t₁.1 + t₁.2 ≤ t₂.1 ∨ t₂.1 + t₂.2 ≤ t₁.1 := by
  have hp := tiles_pairwise ts n lo
  generalize tiles ts lo n = l at h₁ h₂ hp
  induction l with
  | nil => simp at h₁
  | cons x l ih =>
    obtain ⟨hx, hl⟩ := List.pairwise_cons.1 hp
    rcases List.mem_cons.1 h₁ with e₁ | m₁ <;> rcases List.mem_cons.1 h₂ with e₂ | m₂
    · exact absurd (e₁.trans e₂.symm) hne
    · exact .inl (e₁ ▸ hx _ m₂)
    · exact .inr (e₂ ▸ hx _ m₁)
    · exact ih m₁ m₂ hl

/-- The cells of two tiles of an output loop are disjoint. -/
theorem cells_disjoint {ro d : Nat} {b : Box} {t₁ t₂ : Tile} (hd : d < ro) (hdb : d < b.length)
    (h : t₁.1 + t₁.2 ≤ t₂.1 ∨ t₂.1 + t₂.2 ≤ t₁.1) (c : Point) :
    ¬(c ∈ points ((b.set d t₁).take ro) ∧ c ∈ points ((b.set d t₂).take ro)) := by
  rintro ⟨h1, h2⟩
  rw [List.take_set] at h1 h2
  have hlen : d < (b.take ro).length := by simp; omega
  have b1 := (mem_points_bounds h1).2 d
  have b2 := (mem_points_bounds h2).2 d
  rw [boxLo_set hlen, boxSz_set hlen] at b1 b2
  simp only [ite_true, List.length_set] at b1 b2
  have := b1.2 hlen
  have := b2.2 hlen
  omega

/-- A `Grid` or `Threads` split of an output loop: in whatever order its tiles run, the output is the one of the serial order.
With `Op.exec_eq_leaf` for the body of each tile, this is the serial result of the tree. -/
theorem Op.par_tiles (op : Op β α) {ro d : Nat} (rb b : Box) (ts lo n : Nat) (dst : Point → α) (hd : d < ro)
    (hdb : d < b.length) (order : List Tile) (h : order.Perm (tiles ts lo n)) :
    order.foldl (fun dst tl => op.leaf ro rb (b.set d tl) dst) dst =
      (tiles ts lo n).foldl (fun dst tl => op.leaf ro rb (b.set d tl) dst) dst := by
  refine h.foldl_eq' (fun x hx y hy z => ?_) dst
  by_cases hxy : x = y
  · rw [hxy]
  · exact op.leaf_comm ro rb _ _ z
      (cells_disjoint hd hdb (tiles_disjoint (h.mem_iff.1 hy) (h.mem_iff.1 hx) fun e => hxy e.symm))

/-- `Strided`: `n` work-items that each take every `n`-th element, `k` of them, visit the `n * k` elements once. -/
theorem strided_perm (n : Nat) : ∀ k, ((List.range n).flatMap fun i => (List.range k).map fun j => i + j * n).Perm
    (List.range (n * k))
  | 0 => by simp
  | k + 1 => by
    have he : (fun i => (List.range (k + 1)).map fun j => i + j * n) =
        fun i => ((List.range k).map fun j => i + j * n) ++ [i + k * n] := by
      funext i
      simp [List.range_succ]
    rw [he, Nat.mul_succ, List.range_add]
    refine (flatMap_append_perm _ _ _).trans ((strided_perm n k).append ?_)
    have : ((List.range n).flatMap fun i => [i + k * n]) = (List.range n).map fun i => n * k + i := by
      generalize List.range n = l
      induction l with
      | nil => rfl
      | cons i l ih =>
        simp only [List.flatMap_cons, List.map_cons, ih, List.singleton_append]
        rw [Nat.mul_comm k n, Nat.add_comm]
    rw [this]

end Sched
