import Ring.Invariant

namespace Ring.Timeline

/-- Instruction and device event times in one finite interleaving. An event
    which has not occurred has no timestamp. No eventual completion is required. -/
abbrev Event := Option Nat

def By (e : Event) (cut : Nat) := ∃ t, e = some t ∧ t ≤ cut

/-- Enabling edge: executing b requires a strictly earlier a. -/
def Edge (a b : Event) := ∀ t, b = some t → ∃ u, a = some u ∧ u < t

infix:50 " ≺ " => Edge

theorem Edge.trans (ab : a ≺ b) (bc : b ≺ c) : a ≺ c := by
  intro t h
  obtain ⟨u, hu, ut⟩ := bc t h
  obtain ⟨s, hs, su⟩ := ab u hu
  exact ⟨s, hs, by omega⟩

theorem Edge.byTime (ab : a ≺ b) (hb : By b cut) : By a cut := by
  obtain ⟨t, ht, bound⟩ := hb
  obtain ⟨s, hs, lt⟩ := ab t ht
  exact ⟨s, hs, by omega⟩

theorem By.mono (h : By e t) (le : t ≤ u) : By e u := by
  obtain ⟨s, hs, _⟩ := h
  exact ⟨s, hs, by omega⟩

/-- Iterated per-participant or barrier order, restricted to the emitted prefix. -/
theorem chain {f : Nat → Event} (next : ∀ k, k + 1 < n → f k ≺ f (k+1))
    (lt : i < j) (bound : j < n) : f i ≺ f j := by
  induction j with
  | zero => omega
  | succ j ih =>
    by_cases eq : i = j
    · subst i; exact next j bound
    · exact (ih (by omega) (by omega)).trans (next j bound)

theorem circle_gap (_hd : 0 < d) (slot : i % d = j % d) (lt : i < j) : i + d ≤ j := by
  apply Classical.byContradiction
  intro h
  have := window_unique (lo := i) ⟨Nat.le_refl i, by omega⟩ ⟨by omega, by omega⟩ slot
  omega

/-- A writer owns a disjoint slice of each physical slot. Starting a copy
    invalidates that slice until its device completion. The latest issued
    ticket determines the source; all issued writes on the slice must be idle. -/
structure Writes where
  count : Nat
  slot : Nat → Nat
  source : Nat → Nat
  start : Nat → Event
  finish : Nat → Event

def Writes.Idle (w : Writes) (s cut : Nat) : Prop :=
  ∀ k, k < w.count → w.slot k = s → By (w.start k) cut → By (w.finish k) cut

def Writes.Shows (w : Writes) (s ticket block cut : Nat) : Prop :=
  ticket < w.count ∧ w.slot ticket = s ∧ w.source ticket = block ∧
  By (w.start ticket) cut ∧ By (w.finish ticket) cut ∧
  (∀ k, k < w.count → w.slot k = s → By (w.start k) cut → k ≤ ticket) ∧ w.Idle s cut

theorem Writes.shows_of_bounds (w : Writes) (ticket : j < w.count) (slot : w.slot j = s)
    (source : w.source j = block) (issued : w.start j ≺ w.finish j)
    (older : ∀ k, k ≤ j → w.slot k = s → By (w.finish k) cut)
    (newer : ∀ k, j < k → k < w.count → w.slot k = s → ¬ By (w.start k) cut) :
    w.Shows s j block cut := by
  have done_ := older j (Nat.le_refl _) slot
  have last (k) (bound : k < w.count) (same : w.slot k = s) (started : By (w.start k) cut) :
      k ≤ j := by
    apply Classical.byContradiction
    intro h
    exact newer k (by omega) bound same started
  exact ⟨ticket, slot, source, issued.byTime done_, done_, last,
    fun k bound same started => older k (last k bound same started) same⟩

end Ring.Timeline
