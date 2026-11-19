import Ring.Certificate

namespace Ring

inductive Lag where
  | zero | one

def Lag.keep : Lag → Nat
  | .zero => 0
  | .one => 1

def previous : Nat → Option Nat
  | 0 => none
  | n + 1 => some n

def seq (f : Nat → List Op) : Nat → List Op
  | 0 => []
  | n + 1 => seq f n ++ f n

def prologue (d : Nat) := seq (fun j => [.fill j (j % d)]) (d - 1)

/-- Direct calls of the skeleton can prime nonexistent source blocks. -/
theorem prologue_extra_fill (short : count ≤ j) (primed : j < d - 1) :
    count ≤ j ∧ .fill j (j % d) ∈ prologue d := by
  refine ⟨short, ?_⟩
  unfold prologue
  generalize d - 1 = n at primed ⊢
  induction n with
  | zero => omega
  | succ n ih =>
    simp only [seq, List.mem_append, List.mem_singleton]
    by_cases eq : j = n
    · exact Or.inr (by rw [eq])
    · exact Or.inl (ih (by omega))

def refill (d count i : Nat) : List Op :=
  let j := i + (d - 1)
  [.fence, .release (previous i)] ++ (if j < count then [.fill j (j % d)] else [])

def compute (d : Nat) (lag : Lag) (width i : Nat) : List Op :=
  .wgmmaFence :: (List.replicate width (.mma i (i % d)) ++ [.commit, .waitGroups lag.keep])

def iteration (d count : Nat) (lag : Lag) (width : Nat → Nat) (i : Nat) : List Op :=
  .poll i (i % d) :: (match lag with
    | .zero => refill d count i ++ (compute d .zero (width i) i ++ [.next])
    | .one => compute d .one (width i) i ++ (refill d count i ++ [.next]))

def fold (d count : Nat) (lag : Lag) (width : Nat → Nat) : List Op :=
  prologue d ++ seq (iteration d count lag width) count ++ [.waitGroups 0]

-- The source primes d - 1 unconditionally. max accounts for count < d - 1.
private def frontier (d count i : Nat) := min (max (d - 1) count) (i + (d - 1))

private def boundary (d count : Nat) (lag : Lag) (i : Nat) : Mark :=
  ⟨i, frontier d count i, i, i, i - lag.keep, 0⟩

def finalMark (d count : Nat) : Mark :=
  ⟨count, max (d - 1) count, count, count, count, 0⟩

private theorem primes (n : Nat) (bound : n ≤ d) :
    Code d zero (seq (fun j => [.fill j (j % d)]) n) { zero with filled := n } := by
  induction n with
  | zero => exact .nil
  | succ n ih =>
    exact (ih (by omega)).append (.cons (.fill (by dsimp [zero]; omega)
      (by dsimp [zero]; omega)) .nil)

private theorem issues (n : Nat) (lo : v.lo ≤ v.k) (seen : v.k < v.seen) :
    Code d v (List.replicate n (.mma v.k (v.k % d))) { v with batch := v.batch + n } := by
  induction n with
  | zero => exact .nil
  | succ n ih =>
    rw [List.replicate_succ']
    exact ih.append (.cons (.mma lo seen) .nil)

private theorem computes (n : Nat) (lag : Lag) (lo : v.lo ≤ v.k)
    (seen : v.k < v.seen) (bound : v.floor ≤ v.k) :
    Code d v (compute d lag n v.k)
      { v with batch := 0, k := v.k + 1, floor := v.k + 1 - lag.keep } :=
  .cons .wgmmaFence ((issues n lo seen).append (.cons (.commit bound) (.cons .waitGroups .nil)))

private theorem refills (hd : 2 ≤ d) (i : Nat) (lo : v.lo = i)
    (front : v.filled = frontier d count i) (floor : i ≤ v.floor) (k : i ≤ v.k) :
    Code d v (refill d count i) { v with filled := frontier d count (i + 1) } := by
  have retired (j : Nat) (h : previous i = some j) : j < v.floor ∧ j < v.k := by
    cases i <;> simp_all [previous] <;> omega
  unfold refill
  by_cases more : i + (d - 1) < count
  · have now : v.filled = i + (d - 1) := by unfold frontier at front; omega
    have later : frontier d count (i + 1) = v.filled + 1 := by unfold frontier; omega
    simp only [more, ite_true, later]
    rw [← now]
    exact .cons .fence (.cons (.release retired) (.cons (.fill (by omega) (by intro h; constructor <;> omega)) .nil))
  · have later : frontier d count (i + 1) = v.filled := by unfold frontier at *; omega
    simp only [more, ite_false, List.append_nil, later]
    exact .cons .fence (.cons (.release retired) .nil)

private theorem oneIteration (hd : 2 ≤ d) (here : i < count) :
    Code d (boundary d count lag i) (iteration d count lag width i)
      (boundary d count lag (i + 1)) := by
  have issued : i < frontier d count i := by unfold frontier; omega
  cases lag with
  | zero =>
    apply Code.cons (.poll issued)
    apply Code.append (refills hd i rfl rfl (by simp [boundary, Lag.keep]) (Nat.le_refl _))
    apply Code.append (computes (width i) .zero (Nat.le_refl _) (by dsimp [boundary]; omega)
      (by simp [boundary, Lag.keep]))
    exact .cons (.next (by dsimp [boundary]; omega)) .nil
  | one =>
    apply Code.cons (.poll issued)
    apply Code.append (computes (width i) .one (Nat.le_refl _) (by dsimp [boundary]; omega)
      (by dsimp [boundary, Lag.keep]; omega))
    apply Code.append (refills hd i rfl rfl (by dsimp [boundary, Lag.keep]; omega)
      (by dsimp [boundary]; omega))
    exact .cons (.next (by dsimp [boundary]; omega)) .nil

theorem foldCertificate (hd : 2 ≤ d) :
    Code d zero (fold d count lag width) (finalMark d count) := by
  have start : Code d zero (prologue d) (boundary d count lag 0) := by
    have p := primes (d - 1) (d := d) (by omega)
    simpa [prologue, boundary, frontier, zero, Nat.min_eq_right (Nat.le_max_left _ _)] using p
  have loop (n : Nat) (bound : n ≤ count) :
      Code d (boundary d count lag 0) (seq (iteration d count lag width) n)
        (boundary d count lag n) := by
    induction n with
    | zero => exact .nil
    | succ n ih => exact (ih (by omega)).append (oneIteration hd (by omega))
  apply (start.append (loop count (Nat.le_refl _))).append
  apply Code.cons .waitGroups
  have last : frontier d count count = max (d - 1) count := by unfold frontier; omega
  simp only [boundary, last, finalMark, Nat.sub_zero]
  exact .nil

end Ring
