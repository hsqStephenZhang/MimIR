import Ring.Timeline

namespace Ring.Timeline

/-- Count actual publications on a physical barrier in a finite prefix. -/
def tally (p : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n+1 => tally p n + (if p n then 1 else 0)

def occurred (e : Event) (cut : Nat) : Bool := e.any (· ≤ cut)

theorem occurred_iff : occurred e cut = true ↔ By e cut := by
  cases e <;> simp [occurred, By]

theorem tally_prefix {p q : Nat → Bool} (hp : ∀ k, k < n →
    (p k = true ↔ k < i ∧ q k = true)) : tally p n = tally q (min n i) := by
  induction n with
  | zero => simp [tally]
  | succ n ih =>
    have prev := ih (fun k hk => hp k (by omega))
    have atN := hp n (by omega)
    by_cases h : n < i
    · have eq : p n = q n := by cases a : p n <;> cases b : q n <;> simp_all
      simp [tally, prev, eq, Nat.min_eq_left (by omega : n+1 ≤ i), Nat.min_eq_left (by omega : n ≤ i)]
    · have no : p n = false := by cases eq : p n <;> simp_all
      simp [tally, prev, no, Nat.min_eq_right (by omega : i ≤ n+1), Nat.min_eq_right (by omega : i ≤ n)]

theorem visits (hd : 0 < d) (hs : s < d) (n) :
    tally (fun j => decide (j % d = s)) n = n/d + if s < n%d then 1 else 0 := by
  induction n with
  | zero => simp [tally]
  | succ n ih =>
    have rem := Nat.mod_lt n hd
    have decomp := Nat.mod_add_div n d
    by_cases wrap : n%d + 1 = d
    · have eq : n+1 = d*(n/d+1) := by rw [Nat.mul_succ]; omega
      have div := congrArg (· / d) eq
      have mod := congrArg (· % d) eq
      rw [Nat.mul_div_cancel_left _ hd] at div
      rw [Nat.mul_mod_right] at mod
      simp only [tally, ih, div, mod, Nat.not_lt_zero, ite_false, Nat.add_zero, decide_eq_true_eq]
      split <;> split <;> omega
    · have eq : n+1 = (n%d+1) + d*(n/d) := by omega
      have div := congrArg (· / d) eq
      have mod := congrArg (· % d) eq
      have small : n%d+1 < d := by omega
      rw [Nat.add_mul_div_left _ _ hd, Nat.div_eq_of_lt small, Nat.zero_add] at div
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt small] at mod
      simp only [tally, ih, div, mod, decide_eq_true_eq]
      split <;> split <;> split <;> omega

def phaseAt (barrier : Nat → Nat) (publish : Nat → Event) (n s cut : Nat) :=
  tally (fun j => decide (barrier j = s) && occurred (publish j) cut) n

/-- If exactly the earlier visits have published, the physical counter at
    block i's slot equals its unbounded generation number. -/
theorem phase_before (hd : 0 < d) (bound : i ≤ n)
    (hp : ∀ j, j < n → (barrier j = i%d ∧ By (publish j) cut ↔ j < i ∧ j%d = i%d)) :
    phaseAt barrier publish n (i%d) cut = i/d := by
  unfold phaseAt
  rw [tally_prefix (q := fun j => decide (j % d = i % d)) (i := i), Nat.min_eq_right bound,
    visits hd (Nat.mod_lt _ hd)]
  · simp
  · simpa [occurred_iff] using hp

end Ring.Timeline
