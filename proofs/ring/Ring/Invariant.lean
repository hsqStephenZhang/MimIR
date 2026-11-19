import Ring.Machine

namespace Ring

structure Mark where
  lo : Nat        -- Current loop iteration; the physical window is [lo, lo + d).
  filled : Nat    -- Next fill number.
  seen : Nat      -- Next block whose parity has not been acquired.
  k : Nat         -- Next group number.
  floor : Nat     -- Groups strictly below this number have retired.
  batch : Nat     -- Number of pending MMA instructions.

def Before (d j : Nat) (c : Cell) := c.phase = j / d ∧ c.tx = none

def Ready (d j : Nat) (c : Cell) :=
  c.phase = j / d + 1 ∧ c.visible = some (ref d j) ∧ c.tx = none

def Issued (d j : Nat) (c : Cell) :=
  Ready d j c ∨ (c.phase = j / d ∧ ∃ t, c.tx = some t ∧ t.copy.src = ref d j)

def At (d : Nat) (v : Mark) (j : Nat) (c : Cell) : Prop :=
  if j < v.seen then Ready d j c else if j < v.filled then Issued d j c else Before d j c

structure Inv (d : Nat) (v : Mark) (m : Machine) : Prop where
  window : ∀ j, v.lo ≤ j → j < v.lo + d → At d v j (m.slots (j % d))
  commits : m.commits = v.k
  pending : m.pending = List.replicate v.batch (ref d v.k)
  groups : ∀ g ∈ m.groups, v.floor ≤ g.serial ∧ g.serial < v.k ∧
    ∀ r ∈ g.reads, r = ref d g.serial
  ordered : v.lo ≤ v.seen ∧ v.seen ≤ v.filled ∧ v.filled ≤ v.lo + d

def zero : Mark := ⟨0, 0, 0, 0, 0, 0⟩

theorem Inv.initial : Inv d zero initial := by
  refine ⟨?_, rfl, rfl, ?_, by simp [zero]⟩
  · intro j _ hj
    change j < 0 + d at hj
    simp [At, zero, Before, Ring.initial, empty, Nat.div_eq_of_lt (by omega : j < d)]
  · simp [Ring.initial]

private theorem stable_none (h : Device c c') (hn : c.tx = none) : False := by
  cases h <;> simp_all

theorem Issued.hardware (h : Issued d j c) (step : Device c c') : Issued d j c' := by
  rcases h with ready | ⟨phase, t, ht, src⟩
  · exact (stable_none step ready.2.2).elim
  cases step with
  | arrive h _ => cases h.symm.trans ht; exact Or.inr ⟨phase, _, rfl, src⟩
  | deliver h _ _ => cases h.symm.trans ht; exact Or.inr ⟨phase, _, rfl, src⟩
  | publish h _ _ =>
    cases h.symm.trans ht
    exact Or.inl ⟨by simp [phase], congrArg some src, rfl⟩

theorem At.hardware (h : At d v j c) (step : Device c c') : At d v j c' := by
  by_cases seen : j < v.seen
  · simp only [At, seen, ite_true] at h
    exact (stable_none step h.2.2).elim
  by_cases issued : j < v.filled <;> simp [At, seen, issued] at h ⊢
  · exact Issued.hardware h step
  · exact (stable_none step h.2).elim

theorem Inv.hardware (h : Inv d v m) (step : HW m m') : Inv d v m' := by
  cases step with
  | slot step =>
    refine { h with window := ?_ }
    intro j lo hi
    simp only [Machine.put]
    split
    · rename_i eq
      exact (h.window j lo hi).hardware (eq ▸ step)
    · exact h.window j lo hi
  | retire split =>
    refine { h with groups := ?_ }
    intro g hg
    apply h.groups g
    rw [split]
    simp only [List.mem_append, List.mem_cons] at *
    exact hg.imp_right Or.inr

-- Within one moving window the physical slot uniquely identifies the block.
theorem window_unique {lo d x y : Nat} (hx : lo ≤ x ∧ x < lo + d)
    (hy : lo ≤ y ∧ y < lo + d) (eq : x % d = y % d) : x = y := by
  have ax := Nat.mod_add_div x d
  have ay := Nat.mod_add_div y d
  rcases Nat.lt_trichotomy (x / d) (y / d) with lt | same | gt
  · have := Nat.mul_le_mul_left d lt
    rw [Nat.mul_succ] at this
    omega
  · rw [same, eq] at ax
    omega
  · have := Nat.mul_le_mul_left d gt
    rw [Nat.mul_succ] at this
    omega

inductive Rule (d : Nat) : Mark → Op → Mark → Prop
  | fill (space : v.filled < v.lo + d)
      (old : d ≤ v.filled → v.filled - d < v.floor ∧ v.filled - d < v.k) :
      Rule d v (.fill v.filled) { v with filled := v.filled + 1 }
  | poll (issued : v.seen < v.filled) :
      Rule d v (.poll v.seen) { v with seen := v.seen + 1 }
  | mma (inWindow : v.lo ≤ v.k) (acquired : v.k < v.seen) :
      Rule d v (.mma v.k) { v with batch := v.batch + 1 }
  | commit (bound : v.floor ≤ v.k) : Rule d v .commit { v with k := v.k + 1, batch := 0 }
  | waitGroups : Rule d v (.waitGroups n) { v with floor := v.k - n }
  | release (retired : ∀ j, prev = some j → j < v.floor ∧ j < v.k) :
      Rule d v (.release prev) v
  | next (seen : v.lo < v.seen) :
      Rule d v .next { v with lo := v.lo + 1 }

theorem Inv.pollWindow (h : Inv d v m) (issued : v.seen < v.filled) :
    Issued d v.seen (m.slots (v.seen % d)) := by
  have ord := h.ordered
  simpa [At, issued] using h.window v.seen (by omega) (by omega)

theorem Inv.pollReady (h : Inv d v m) (issued : v.seen < v.filled)
    (ok : passes d v.seen m) : Ready d v.seen (m.slots (v.seen % d)) := by
  rcases h.pollWindow issued with ready | ⟨phase, _⟩
  · exact ready
  · exact (ok (congrArg (· % 2) phase)).elim

theorem Inv.preserve (h : Inv d v m) (rule : Rule d v op w)
    (step : Host d bytes op m m') : Inv d w m' := by
  have ord := h.ordered
  cases rule with
  | fill space old =>
    cases step
    refine ⟨?_, h.commits, h.pending, h.groups, by dsimp; omega⟩
    intro j lo hi
    dsimp only at lo hi
    have win := h.window j lo hi
    by_cases eq : j = v.filled
    · subst j
      have before : Before d v.filled (m.slots (v.filled % d)) := by
        simpa [At, show ¬ v.filled < v.seen by omega] using win
      simp only [At, Machine.put, show ¬ v.filled < v.seen by omega,
        ite_false, Nat.lt_succ_self, ite_true]
      exact Or.inr ⟨before.1, _, rfl, rfl⟩
    · have diff : j % d ≠ v.filled % d :=
        fun e => eq (window_unique ⟨lo, hi⟩ ⟨by omega, space⟩ e)
      simpa [At, Machine.put, diff, show (j < v.filled + 1) = (j < v.filled) by
        apply propext; omega] using win
  | poll issued =>
    cases step with
    | poll ok =>
      refine { h with window := ?_, ordered := by dsimp; omega }
      intro j lo hi
      by_cases eq : j = v.seen
      · subst j; simpa [At] using h.pollReady issued ok
      · simpa [At, show (j < v.seen + 1) = (j < v.seen) by apply propext; omega]
          using h.window j lo hi
  | mma _ _ =>
    cases step
    exact { h with pending := by simp [List.replicate_succ, h.pending] }
  | commit bound =>
    cases step
    refine ⟨h.window, by simp [h.commits], rfl, ?_, ord⟩
    intro g hg
    simp only [List.mem_cons] at hg
    rcases hg with rfl | hg
    · simp only [h.commits]
      refine ⟨?_, by omega, ?_⟩
      · exact bound
      · intro r hr
        have := List.eq_of_mem_replicate (h.pending ▸ hr)
        exact this
    · obtain ⟨bound, lt, reads⟩ := h.groups _ hg
      exact ⟨bound, by dsimp; omega, reads⟩
  | waitGroups =>
    cases step with
    | waitGroups recent =>
      refine { h with groups := ?_ }
      intro g hg
      have := recent g hg
      rw [h.commits] at this
      exact ⟨by dsimp; omega, (h.groups _ hg).2⟩
  | release => cases step; exact h
  | next seen =>
    cases step
    refine { h with window := ?_, ordered := by dsimp; omega }
    intro j lo hi
    dsimp at lo hi
    by_cases edge : j = v.lo + d
    · subst j
      have hd : 0 < d := by omega
      have ready : Ready d v.lo (m.slots (v.lo % d)) := by
        simpa [At, seen] using h.window v.lo (by omega) (by omega)
      simp [At, show ¬ v.lo + d < v.seen by omega, show ¬ v.lo + d < v.filled by omega,
        Before, Nat.add_div_right _ hd, ready.1, ready.2.2]
    · exact h.window j (by omega) (by omega)

end Ring
