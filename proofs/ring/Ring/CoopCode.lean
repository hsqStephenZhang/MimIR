import Ring.Cooperative
import Ring.TmaCode

set_option linter.unusedSimpArgs false

namespace Ring.Cooperative
open Timeline Tma

/-- Local rules of one consumer group of the cooperative TMA fold (`lag = 1`, `exact`).
    The `leader` also initializes the full barriers and issues every fill. -/
def rules (d count : Nat) (leader : Bool) : Rules := fun s prev x => match x with
  | .init b n =>
    [("init outside the leader's set-up", leader && s .barrier == 0),
     ("init is not full I with one arrival, in order", decide (s .init < d ∧ b = .full (s .init) ∧ n = 1))]
  | .barrier =>
    [("set-up barrier before all full barriers are initialized",
       decide (s .barrier = 0 → leader = true → s .init = d)),
     ("step barrier not after the step's wgmma_wait",
       decide (0 < s .barrier → s .barrier ≤ count ∧ s .fwait = s .barrier ∧ s .bwait = s .barrier))]
  | .expectTx b _ =>
    [("expect_tx outside the leader", leader),
     ("more fills than blocks", decide (s .expect < count)),
     ("expect_tx barrier is not full J % d", decide (b = .full (s .expect % d))),
     ("prologue fill out of place", decide (s .expect < d - 1 → s .barrier = 1 ∧ s .fwait = 0)),
     ("refill without the step's CTA barrier", decide (d - 1 ≤ s .expect →
       s .barrier = s .expect - (d - 1) + 2 ∧ s .fwait = s .expect - (d - 1) + 1))]
  | .tmaLoad sl b src _ =>
    [("tma_load outside the leader", leader),
     ("tma_load outside a fill", inFill prev && decide (0 < s .expect)),
     ("tma_load slot is not J % d", decide (sl = (s .expect - 1) % d)),
     ("tma_load barrier is not full J % d", decide (b = .full ((s .expect - 1) % d))),
     ("tma_load source is not block J", decide (src = s .expect - 1))]
  | .mbarWait b p =>
    [("more waits than blocks", decide (s .fwait < count)),
     ("mbar_wait before the previous step's barrier",
       decide (s .barrier = s .fwait + 1 ∧ s .bfence = s .fwait ∧ s .bwait = s .fwait)),
     ("mbar_wait before the fills it needs", decide (leader = true → s .expect = min count (d - 1 + s .fwait))),
     ("mbar_wait barrier is not full I % d", decide (b = .full (s .fwait % d))),
     ("mbar_wait parity is not (I / d) % 2", decide (p = s .fwait / d % 2))]
  | .wgmmaFence =>
    [("wgmma_fence out of place", decide (s .bfence + 1 = s .fwait ∧ s .bcommit = s .bfence ∧
       s .bwait = s .bfence ∧ s .barrier = s .fwait))]
  | .read b sl =>
    [("read of another block than the step's", decide (b + 1 = s .fwait)),
     ("read outside the step's open batch", decide (s .bfence = s .fwait ∧ s .bcommit + 1 = s .bfence)),
     ("read slot is not block % d", decide (sl = b % d))]
  | .wgmmaCommit => [("wgmma_commit without an open batch", decide (s .bcommit + 1 = s .bfence))]
  | .wgmmaWait n =>
    [("wgmma_wait out of place", decide (if s .bwait < count then
        n = 1 ∧ s .bcommit = s .bwait + 1 ∧ s .barrier = s .bwait + 1
      else s .bwait = count ∧ n = 0 ∧ s .bcommit = count ∧ s .barrier = count + 1))]
  | .mbarArrive _ => [("mbar_arrive in a cooperative ring", false)]
  | .settle =>
    [("settle out of place", decide (s .bwait = count + 1 ∧ s .fwait = count ∧
       (leader = true → s .expect = count)))]

/-- The checker for one consumer group's program-order sequence. -/
def accepts (d count : Nat) (leader : Bool) (P : List Instr) : Bool :=
  decide (2 ≤ d ∧ d - 1 ≤ count) && run (rules d count leader) P && bytesMatch P

/-! ### The canonical sequence -/

def fillOps (d loads bytes j : Nat) : List Instr :=
  .expectTx (.full (j % d)) (loads * bytes) :: List.replicate loads (.tmaLoad (j % d) (.full (j % d)) j bytes)

def initOps : Nat → List Instr
  | 0 => []
  | n + 1 => initOps n ++ [.init (.full n) 1]

def primeOps (d loads bytes : Nat) : Nat → List Instr
  | 0 => []
  | j + 1 => primeOps d loads bytes j ++ fillOps d loads bytes j

def stepOps (d count : Nat) (leader : Bool) (loads bytes mmas i : Nat) : List Instr :=
  .mbarWait (.full (i % d)) (i / d % 2) :: .wgmmaFence :: (List.replicate mmas (.read i (i % d)) ++
  .wgmmaCommit :: .wgmmaWait 1 :: .barrier ::
  (if leader && decide (i + (d - 1) < count) then fillOps d loads bytes (i + (d - 1)) else []))

def stepsOps (d count : Nat) (leader : Bool) (loads bytes mmas : Nat) : Nat → List Instr
  | 0 => []
  | i + 1 => stepsOps d count leader loads bytes mmas i ++ stepOps d count leader loads bytes mmas i

def setupOps (d : Nat) (leader : Bool) (loads bytes : Nat) : List Instr :=
  if leader then initOps d ++ [.barrier] ++ primeOps d loads bytes (d - 1) else [.barrier]

/-- Each fill copies `loads` boxes of `bytes` each; each step issues `mmas` reads in one batch. -/
def expected (d count : Nat) (leader : Bool) (loads bytes mmas : Nat) : List Instr :=
  setupOps d leader loads bytes ++ stepsOps d count leader loads bytes mmas count ++ [.wgmmaWait 0, .settle]

private def state (leader : Bool) (d count i : Nat) : Counts
  | .init => if leader then d else 0
  | .expect => if leader then min count (d - 1 + i) else 0
  | .barrier => i + 1
  | .fwait | .bfence | .bcommit | .bwait => i
  | _ => 0

private def inits (n : Nat) : Counts
  | .init => n
  | _ => 0

private def primed (d j : Nat) : Counts
  | .init => d
  | .barrier => 1
  | .expect => j
  | _ => 0

section
variable {d count : Nat} {leader : Bool}

private theorem steps_inits (hn : n ≤ d) :
    Steps (rules d count true) (counts []) (inits n) (initOps n) := by
  induction n with
  | zero => exact Steps.nil.eq (by funext k; cases k <;> rfl)
  | succ n ih =>
    refine (ih (by omega)).append ((Steps.single ?_ rfl rfl).eq ?_)
    · intro p _; simp [rules, inits]; omega
    · funext k; cases k <;> simp [inits, Counts.add, Instr.kind?]

private theorem steps_prime (hd : 2 ≤ d) (hc : d - 1 ≤ count) (hl : 0 < loads) (hj : j ≤ d - 1) :
    Steps (rules d count true) (primed d 0) (primed d j) (primeOps d loads bytes j) := by
  induction j with
  | zero => exact Steps.nil
  | succ j ih =>
    refine (ih (by omega)).append ((Steps.fill ?_ ?_ rfl hl).eq ?_)
    · have : j < count := by omega
      intro p _; simp [rules, primed, this]; omega
    · intro p hp; simp [rules, primed, Counts.add, Instr.kind?, hp]
    · funext k; cases k <;> simp [primed, Counts.add, Instr.kind?]

private theorem steps_setup (hd : 2 ≤ d) (hc : d - 1 ≤ count) (hl : 0 < loads) :
    Steps (rules d count leader) (counts []) (state leader d count 0) (setupOps d leader loads bytes) := by
  cases leader with
  | false =>
    refine (Steps.single ?_ rfl rfl).eq ?_
    · intro p _; simp [rules]
    · funext k; cases k <;> simp [state, Counts.add, Instr.kind?, counts]
  | true =>
    have bar : Steps (rules d count true) (inits d) (primed d 0) [.barrier] := by
      refine (Steps.single ?_ rfl rfl).eq ?_
      · intro p _; simp [rules, inits]
      · funext k; cases k <;> simp [primed, inits, Counts.add, Instr.kind?]
    refine ((steps_inits (Nat.le_refl d)).append bar |>.append (steps_prime hd hc hl (Nat.le_refl _))).eq ?_
    funext k; cases k <;> simp [primed, state] <;> omega

private theorem steps_step (hi : i < count) (hl : 0 < loads) :
    Steps (rules d count leader) (state leader d count i) (state leader d count (i + 1))
      (stepOps d count leader loads bytes mmas i) := by
  unfold stepOps
  refine Steps.cons ?_ (by simp [state]) rfl (Steps.cons ?_ (by simp [state, Counts.add, Instr.kind?]) rfl
    ((Steps.replicate ?_ (by simp [state, Counts.add, Instr.kind?]) rfl rfl).append
    (Steps.cons ?_ (by simp [state, Counts.add, Instr.kind?]) rfl
    (Steps.cons ?_ (by simp [state, Counts.add, Instr.kind?]) rfl
    (Steps.cons ?_ (by simp [state, Counts.add, Instr.kind?]) rfl ?_)))))
  · intro p _; cases leader <;> simp [rules, state, hi] <;> omega
  · intro p _; simp [rules, state, Counts.add, Instr.kind?]
  · intro p _; simp [rules, state, Counts.add, Instr.kind?]
  · intro p _; simp [rules, state, Counts.add, Instr.kind?]
  · intro p _; simp [rules, state, Counts.add, Instr.kind?, hi]
  · intro p _; simp [rules, state, Counts.add, Instr.kind?]; omega
  · by_cases refill : leader = true ∧ i + (d - 1) < count
    · obtain ⟨rfl, lt⟩ := refill
      simp only [Bool.true_and, lt, decide_true, ite_true, fillOps]
      have e : min count (d - 1 + i) = i + (d - 1) := by omega
      refine (Steps.fill ?_ ?_ (by simp [state, Counts.add, Instr.kind?]) hl).eq ?_
      · intro p _; simp [rules, state, Counts.add, Instr.kind?, e, lt]
      · intro p hp; simp [rules, state, Counts.add, Instr.kind?, hp, e]
      · funext k; cases k <;> simp [state, Counts.add, Instr.kind?] <;> omega
    · have : (leader && decide (i + (d - 1) < count)) = false := by
        cases leader <;> simp_all
      rw [this]
      refine Steps.nil.eq ?_
      funext k; cases k <;> cases leader <;> simp_all [state, Counts.add, Instr.kind?] <;> omega

private theorem cons_plain (h : x.kind? ≠ some .expect) (hl : isLoad x = false)
    (b : Balanced L) : Balanced (x :: L) := by
  rw [← List.singleton_append]; exact (Balanced.plain h hl).append b

private theorem balanced_expected : Balanced (expected d count leader loads bytes mmas) := by
  have inits (n : Nat) : Balanced (initOps n) := by
    induction n with
    | zero => exact Balanced.nil
    | succ n ih => exact ih.append (Balanced.plain (by simp [Instr.kind?]) rfl)
  have prime (j : Nat) : Balanced (primeOps d loads bytes j) := by
    induction j with
    | zero => exact Balanced.nil
    | succ j ih => exact ih.append Balanced.fill
  have step (i : Nat) : Balanced (stepOps d count leader loads bytes mmas i) := by
    unfold stepOps
    refine cons_plain ?_ rfl (cons_plain ?_ rfl ((Balanced.replicate (Balanced.plain ?_ rfl)).append
      (cons_plain ?_ rfl (cons_plain ?_ rfl (cons_plain ?_ rfl ?_))))) <;> try simp [Instr.kind?]
    split
    · exact Balanced.fill
    · exact Balanced.nil
  have steps (n : Nat) : Balanced (stepsOps d count leader loads bytes mmas n) := by
    induction n with
    | zero => exact Balanced.nil
    | succ n ih => exact ih.append (step n)
  have setup : Balanced (setupOps d leader loads bytes) := by
    unfold setupOps
    split
    · exact ((inits d).append (Balanced.plain (by simp [Instr.kind?]) rfl)).append (prime _)
    · exact Balanced.plain (by simp [Instr.kind?]) rfl
  exact (setup.append (steps _)).append
    (cons_plain (by simp [Instr.kind?]) rfl (Balanced.plain (by simp [Instr.kind?]) rfl))

theorem expected_accepted (hd : 2 ≤ d) (hc : d - 1 ≤ count) (hl : 0 < loads) :
    accepts d count leader (expected d count leader loads bytes mmas) = true := by
  have steps (n : Nat) (hn : n ≤ count) :
      Steps (rules d count leader) (state leader d count 0) (state leader d count n)
        (stepsOps d count leader loads bytes mmas n) := by
    induction n with
    | zero => exact Steps.nil
    | succ n ih => exact (ih (by omega)).append (steps_step (by omega) hl)
  have tail : Steps (rules d count leader) (state leader d count count)
      ((state leader d count count).add (.wgmmaWait 0) |>.add .settle) [.wgmmaWait 0, .settle] := by
    refine Steps.cons ?_ (by simp [state]) rfl (Steps.single ?_ (by simp [state, Counts.add, Instr.kind?]) rfl)
    · intro p _; simp [rules, state]
    · intro p _; cases leader <;> simp [rules, state, Counts.add, Instr.kind?] <;> omega
  have all := ((steps_setup (bytes := bytes) hd hc hl).append (steps count (Nat.le_refl _))).append tail
  unfold accepts expected
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨hd, hc⟩, run_of_steps all (by simp [state, Counts.add, Instr.kind?])⟩, bytesMatch_of balanced_expected⟩

end

/-! ### Rules at a position -/

structure Accepted (d count : Nat) (leader : Bool) (P : List Instr) : Prop where
  depth : 2 ≤ d
  prime : d - 1 ≤ count
  obeys : Obeys (rules d count leader) P

theorem accepted (h : accepts d count leader P = true) : Accepted d count leader P := by
  unfold accepts at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, obeys_of_run h.1.2⟩

section
variable {d count : Nat} {leader : Bool} {P : List Instr} (acc : Accepted d count leader P)
include acc

theorem Accepted.expect (h : P[q]? = some (.expectTx b n)) :
    leader = true ∧ counts (P.take q) .expect < count ∧ b = .full (counts (P.take q) .expect % d) ∧
    (counts (P.take q) .expect < d - 1 → counts (P.take q) .barrier = 1 ∧ counts (P.take q) .fwait = 0) ∧
    (d - 1 ≤ counts (P.take q) .expect →
      counts (P.take q) .barrier = counts (P.take q) .expect - (d - 1) + 2 ∧
      counts (P.take q) .fwait = counts (P.take q) .expect - (d - 1) + 1) := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r

theorem Accepted.load (h : P[q]? = some (.tmaLoad sl b src n)) :
    0 < counts (P.take q) .expect ∧ sl = (counts (P.take q) .expect - 1) % d ∧
    b = .full ((counts (P.take q) .expect - 1) % d) ∧ src = counts (P.take q) .expect - 1 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.2.1.2, r.2.2⟩

theorem Accepted.poll (h : P[q]? = some (.mbarWait b p)) :
    counts (P.take q) .fwait < count ∧ counts (P.take q) .barrier = counts (P.take q) .fwait + 1 ∧
    counts (P.take q) .bfence = counts (P.take q) .fwait ∧ counts (P.take q) .bwait = counts (P.take q) .fwait ∧
    (leader = true → counts (P.take q) .expect = min count (d - 1 + counts (P.take q) .fwait)) ∧
    b = .full (counts (P.take q) .fwait % d) ∧ p = counts (P.take q) .fwait / d % 2 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.1, r.2.1.1, r.2.1.2.1, r.2.1.2.2, r.2.2⟩

theorem Accepted.bfence (h : P[q]? = some .wgmmaFence) :
    counts (P.take q) .bfence + 1 = counts (P.take q) .fwait ∧
    counts (P.take q) .bcommit = counts (P.take q) .bfence := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.1, r.2.1⟩

theorem Accepted.read (h : P[q]? = some (.read b sl)) :
    b + 1 = counts (P.take q) .fwait ∧ counts (P.take q) .bfence = counts (P.take q) .fwait ∧
    counts (P.take q) .bcommit + 1 = counts (P.take q) .bfence ∧ sl = b % d := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.1, r.2.1.1, r.2.1.2, r.2.2⟩

theorem Accepted.bcommit (h : P[q]? = some .wgmmaCommit) :
    counts (P.take q) .bcommit + 1 = counts (P.take q) .bfence := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r

theorem Accepted.bwait (h : P[q]? = some (.wgmmaWait n)) :
    if counts (P.take q) .bwait < count then
      n = 1 ∧ counts (P.take q) .bcommit = counts (P.take q) .bwait + 1 ∧
      counts (P.take q) .barrier = counts (P.take q) .bwait + 1
    else counts (P.take q) .bwait = count ∧ n = 0 ∧ counts (P.take q) .bcommit = count ∧
      counts (P.take q) .barrier = count + 1 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, decide_eq_true_eq] at r
  exact r

theorem Accepted.barrier (h : P[q]? = some .barrier) (pos : 0 < counts (P.take q) .barrier) :
    counts (P.take q) .barrier ≤ count ∧ counts (P.take q) .fwait = counts (P.take q) .barrier ∧
    counts (P.take q) .bwait = counts (P.take q) .barrier := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r.2 pos

theorem Accepted.settle (h : P[q]? = some .settle) :
    counts (P.take q) .bwait = count + 1 ∧ counts (P.take q) .fwait = count ∧
    (leader = true → counts (P.take q) .expect = count) := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r

end

section
variable {d count : Nat} {leader : Bool} {P : List Instr} (acc : Accepted d count leader P)
include acc

theorem Accepted.final : ∃ qs, nth .settle P 0 = some qs ∧ counts (P.take qs) .bwait = count + 1 ∧
    counts (P.take qs) .fwait = count ∧ (leader = true → counts (P.take qs) .expect = count) := by
  obtain ⟨qs, hs, hx, -⟩ := acc.obeys.settleAt
  exact ⟨qs, hs, acc.settle hx⟩

theorem Accepted.fills (hl : leader = true) : counts P .expect = count := by
  obtain ⟨qs, hs, -, -, e⟩ := acc.final
  rw [acc.obeys.total hs]; simp [e hl]

theorem Accepted.fill (hl : leader = true) (hj : j < count) :
    ∃ q n m, nth .expect P j = some q ∧ P[q]? = some (.expectTx (.full (j % d)) n) ∧
      P[q + 1]? = some (.tmaLoad (j % d) (.full (j % d)) j m) := by
  obtain ⟨qs, hs, -, -, e⟩ := acc.final
  obtain ⟨q, hq, -⟩ := nth_lt (k := .expect) (n := j) (p := qs) (P := P) (by rw [e hl]; exact hj)
  obtain ⟨x, hx, kind⟩ := nth_instr hq
  obtain ⟨b, n, rfl⟩ := kind_expect kind
  have cnt := (nth_spec hq).2
  have ex := acc.expect hx
  rw [cnt] at ex
  obtain ⟨sl, b', src, m, hy⟩ := acc.obeys.next hx
  have ld := acc.load hy
  rw [counts_at hx] at ld
  simp only [Instr.kind?, ite_true, cnt, Nat.add_sub_cancel] at ld
  obtain ⟨-, rfl, rfl, rfl⟩ := ld
  exact ⟨q, n, m, hq, ex.2.2.1 ▸ hx, hy⟩

theorem Accepted.polled (hi : i < count) :
    ∃ q p, nth .fwait P i = some q ∧ P[q]? = some (.mbarWait (.full (i % d)) p) ∧ p = i / d % 2 ∧
      counts (P.take q) .barrier = i + 1 ∧ (leader = true → counts (P.take q) .expect = min count (d - 1 + i)) := by
  obtain ⟨qs, hs, -, f, -⟩ := acc.final
  obtain ⟨q, hq, -⟩ := nth_lt (k := .fwait) (n := i) (p := qs) (P := P) (by rw [f]; exact hi)
  obtain ⟨x, hx, kind⟩ := nth_instr hq
  obtain ⟨b, p, rfl⟩ := kind_fwait kind
  have cnt := (nth_spec hq).2
  have r := acc.poll hx
  rw [cnt] at r
  obtain ⟨-, bar, -, -, ex, hb, hp⟩ := r
  cases hb
  exact ⟨q, p, hq, hx, hp, bar, ex⟩

theorem Accepted.no_barrier (hk : count < k) : nth .barrier P k = none := by
  cases h : nth .barrier P k with
  | none => rfl
  | some q =>
    obtain ⟨x, hx, kind⟩ := nth_instr h
    rw [kind_barrier kind] at hx
    have := acc.barrier hx (by rw [(nth_spec h).2]; omega)
    rw [(nth_spec h).2] at this
    omega

theorem Accepted.waitAt (h : nth .bwait P i = some q) :
    ∃ n, P[q]? = some (.wgmmaWait n) ∧ (if i < count then n = 1 ∧ counts (P.take q) .bcommit = i + 1 ∧
      counts (P.take q) .barrier = i + 1 else i = count ∧ n = 0 ∧ counts (P.take q) .bcommit = count ∧
      counts (P.take q) .barrier = count + 1) := by
  obtain ⟨x, hx, kind⟩ := nth_instr h
  obtain ⟨n, rfl⟩ := kind_bwait kind
  have r := acc.bwait hx
  rw [(nth_spec h).2] at r
  exact ⟨n, hx, r⟩

end

/-! ### Timelines consistent with program order -/

/-- One timeline per consumer group: program positions, CTA barrier releases,
    the device completion of each fill and of each WGMMA batch. -/
structure Exec (groups : Nat) where
  track : Fin groups → Track
  release : Nat → Event
  arrived : Nat → Event
  landed : Nat → Event
  publish : Nat → Event
  retire : Fin groups → Nat → Event

/-- Program order, the CTA barrier, and the mbarrier and WGMMA hardware rules.
    Each fill's `expect_tx` arrives once; its phase completes after that arrival and after every byte of its
    `tma_load`s landed. A completed phase of a full barrier is one published fill on it. -/
structure Consistent (prog : Fin groups → List Instr) (lead : Fin groups) (e : Exec groups) : Prop where
  order : ∀ g, (e.track g).Mono
  arrive : ∀ g k, (e.track g).at (nth .barrier (prog g) k) ≺ e.release k
  pass : ∀ g k q, nth .barrier (prog g) k = some q → e.release k ≺ (e.track g).at (some (q + 1))
  expectArrive : ∀ j, (e.track lead).at (nth .expect (prog lead) j) ≺ e.arrived j
  loadLanded : ∀ q sl b src n, (prog lead)[q]? = some (.tmaLoad sl b src n) →
    (e.track lead).at (some q) ≺ e.landed (counts ((prog lead).take q) .expect - 1)
  noFill : ∀ j, nth .expect (prog lead) j = none → e.landed j = none
  arrivedPublish : ∀ j, e.arrived j ≺ e.publish j
  landedPublish : ∀ j, e.landed j ≺ e.publish j
  parity : ∀ g q b p t, (prog g)[q]? = some (.mbarWait (.full b) p) → (e.track g).at (some q) = some t →
    phaseAt (fillBar (prog lead)) e.publish (counts (prog lead) .expect) b t % 2 ≠ p
  retire : ∀ g c, (e.track g).at (nth .bcommit (prog g) c) ≺ e.retire g c
  waitGroup : ∀ g q n c, (prog g)[q]? = some (.wgmmaWait n) →
    (e.track g).at (nth .bcommit (prog g) (c + n)) ≺ (e.track g).at (some q) →
    e.retire g c ≺ (e.track g).at (some q)

/-- The model CTA of a cooperative ring, read off the programs and the timeline.
    Group `g`'s representative thread stands for its warpgroup. -/
def cta (count : Nat) (prog : Fin groups → List Instr) (lead : Fin groups) (e : Exec groups) : CTA groups where
  leader := lead
  start j := (e.track lead).at (nth .expect (prog lead) j)
  arrived := e.arrived
  bytesComplete := e.landed
  publish := e.publish
  fullBarrier := fillBar (prog lead)
  fillSlot j := (loadSlot (prog lead) j).getD 0
  source j := (loadSrc (prog lead) j).getD 0
  pollSlot g i := (barOf (prog g) .fwait i).getD 0
  poll g i := (e.track g).at (nth .fwait (prog g) i)
  batchFence g i := (e.track g).at (nth .bfence (prog g) i)
  commit g i := (e.track g).at (nth .bcommit (prog g) i)
  done_ := e.retire
  waitGroup g i := (e.track g).at (if i < count then nth .bwait (prog g) i else none)
  fence i := e.release (i + 1)
  finalWait g := (e.track g).at (nth .bwait (prog g) count)

/-- The read at position `q` of group `g`, if any. -/
def readAt (prog : Fin groups → List Instr) (e : Exec groups) (g : Fin groups) (q : Nat) : Option (Read groups) :=
  match (prog g)[q]? with
  | some (.read b s) => some ⟨g, b, s, (e.track g).at (some q)⟩
  | _ => none

/-- Group `g` is checked as the leader exactly when it is `lead`. -/
abbrev AllAccepted (d count : Nat) (prog : Fin groups → List Instr) (lead : Fin groups) : Prop :=
  ∀ g, Accepted d count (decide (g = lead)) (prog g)

section
variable {d count : Nat} {prog : Fin groups → List Instr} {lead : Fin groups} {e : Exec groups}

theorem fold_of_accepted (acc : AllAccepted d count prog lead) (con : Consistent prog lead e) :
    Fold (cta count prog lead e) d count := by
  have L := acc lead
  simp only [decide_true] at L
  have M := con.order
  have hd := L.depth
  have hc := L.prime
  have tail (g) (q : Nat) (h : nth .bwait (prog g) count = some q) :
      ∃ n, (prog g)[q]? = some (.wgmmaWait n) ∧ n = 0 ∧ counts ((prog g).take q) .bcommit = count := by
    obtain ⟨n, hx, r⟩ := (acc g).waitAt h
    simp only [Nat.lt_irrefl, ite_false] at r
    exact ⟨n, hx, r.2.1, r.2.2.1⟩
  constructor
  · exact L.prime
  · intro j _
    exact (M lead).ordered fun q h => nth_earlier h (by omega)
  · intro j hj
    exact (M lead).ordered fun q h => by
      change nth Kind.fwait (prog lead) _ = some q at h
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, p, rfl⟩ := kind_fwait kind
      have r := (L.poll hx).2.2.2.2.1 rfl
      rw [(nth_spec h).2] at r
      exact nth_lt (by rw [r]; omega)
  · intro i hi
    exact (M lead).ordered fun q h => by
      change nth Kind.fwait (prog lead) _ = some q at h
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, p, rfl⟩ := kind_fwait kind
      have r := (L.poll hx).2.2.2.2.1 rfl
      rw [(nth_spec h).2] at r
      exact nth_lt (by rw [r]; omega)
  · intro j hj
    obtain ⟨q, n, m, hq, hx, hy⟩ := L.fill rfl hj
    simp [cta, loadSlot, loadOf, hq, hy]
  · intro j hj
    obtain ⟨q, n, m, hq, hx, hy⟩ := L.fill rfl hj
    simp [cta, fillBar, barOf_eq hq hx, Bar.slot]
  · intro j hj
    obtain ⟨q, n, m, hq, hx, hy⟩ := L.fill rfl hj
    simp [cta, loadSrc, loadOf, hq, hy]
  · intro g i hi
    obtain ⟨q, p, hq, hx, -⟩ := (acc g).polled hi
    simp [cta, barOf_eq hq hx, Bar.slot]
  · intro j; exact con.expectArrive j
  · intro j
    cases h : nth .expect (prog lead) j with
    | none => simp only [cta, h, con.noFill j h]; exact edge_none_right
    | some q =>
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, n, rfl⟩ := kind_expect kind
      obtain ⟨sl, b', src, m, hy⟩ := L.obeys.next hx
      have land := con.loadLanded _ _ _ _ _ hy
      rw [counts_at hx, (nth_spec h).2] at land
      simp only [Instr.kind?, ite_true, Nat.add_sub_cancel] at land
      simp only [cta, h]
      exact ((M lead).lt (Nat.lt_succ_self q)).trans land
  · intro j; exact con.arrivedPublish j
  · intro j; exact con.landedPublish j
  · intro g i t hi ht
    obtain ⟨q, p, hq, hx, hp, -⟩ := (acc g).polled hi
    have := con.parity g q _ p t hx (by simpa [cta, hq] using ht)
    rw [L.fills rfl] at this
    simpa [cta, phase, barOf_eq hq hx, Bar.slot, hp] using this
  · intro g i
    exact (M g).ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      rw [kind_bfence kind] at hx
      have r := (acc g).bfence hx
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro g i
    exact (M g).ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      rw [kind_bcommit kind] at hx
      have r := (acc g).bcommit hx
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro g i; exact con.retire g i
  · intro g i
    simp only [cta]
    split
    · exact (M g).ordered fun q h => by
        obtain ⟨n, hx, r⟩ := (acc g).waitAt h
        simp only [‹i < count›, ite_true] at r
        exact nth_lt (by omega)
    · exact edge_none_right
  · intro g i j lt
    simp only [cta]
    split
    · rename_i hi
      cases h : nth .bwait (prog g) i with
      | none => exact edge_none_right
      | some q =>
        obtain ⟨n, hx, r⟩ := (acc g).waitAt h
        simp only [hi, ite_true] at r
        obtain ⟨rfl, bc, -⟩ := r
        apply con.waitGroup g q 1 j hx
        exact (M g).ordered fun q' h' => by cases h'; exact nth_lt (by omega)
    · exact edge_none_right
  · intro g i
    simp only [cta]
    have arr := con.arrive g (i + 1)
    split
    · rename_i hi
      cases hb : nth .barrier (prog g) (i + 1) with
      | none =>
        rw [hb] at arr
        rw [edge_none_left arr]; exact edge_none_right
      | some p =>
        rw [hb] at arr
        refine Edge.trans ?_ arr
        exact (M g).ordered fun q h => by
          cases h
          obtain ⟨x, hx, kind⟩ := nth_instr hb
          rw [kind_barrier kind] at hx
          have r := (acc g).barrier hx (by rw [(nth_spec hb).2]; omega)
          rw [(nth_spec hb).2] at r
          exact nth_lt (by omega)
    · rename_i hi
      rw [(acc g).no_barrier (by omega)] at arr
      rw [edge_none_left arr]; exact edge_none_right
  · intro g i hi
    cases h : nth .fwait (prog g) (i + 1) with
    | none => simp only [cta, h]; exact edge_none_right
    | some q =>
      obtain ⟨q', p, hq, hx, -, bar, -⟩ := (acc g).polled hi
      rw [h] at hq; cases hq
      obtain ⟨pb, hb, lt⟩ := nth_lt (k := .barrier) (n := i + 1) (P := prog g) (p := q) (by omega)
      simp only [cta, h]
      exact (M g).later (con.pass g _ pb hb) (by omega)
  · intro i hi
    obtain ⟨q, n, m, hq, hx, -⟩ := L.fill rfl hi
    have ex := (L.expect hx).2.2.2.2
    rw [(nth_spec hq).2] at ex
    obtain ⟨bar, -⟩ := ex (by omega)
    obtain ⟨pb, hb, lt⟩ := nth_lt (k := .barrier) (n := i + 1) (P := prog lead) (p := q) (by omega)
    simp only [cta, hq]
    exact (M lead).later (con.pass lead _ pb hb) (by omega)
  · intro g i hi
    cases h : nth .bwait (prog g) count with
    | none => simp only [cta, h]; exact edge_none_right
    | some q =>
      obtain ⟨n, hx, rfl, bc⟩ := tail g q h
      simp only [cta, h]
      apply con.waitGroup g q 0 i hx
      exact (M g).ordered fun q' h' => by cases h'; exact nth_lt (by omega)

end

section
variable {d count : Nat} {prog : Fin groups → List Instr} {lead : Fin groups} {e : Exec groups}

theorem readCode_of_accepted (acc : AllAccepted d count prog lead) (con : Consistent prog lead e)
    (hr : readAt prog e g q = some r) : ReadCode (cta count prog lead e) d count r := by
  unfold readAt at hr
  split at hr
  case h_2 => cases hr
  rename_i b s hq
  cases hr
  have rr := (acc g).read hq
  have M := con.order g
  refine ⟨?_, rr.2.2.2, ?_, ?_⟩
  · obtain ⟨p, hp, lt⟩ := nth_lt (k := .fwait) (n := b) (P := prog g) (p := q) (by omega)
    obtain ⟨x, hx, kind⟩ := nth_instr hp
    obtain ⟨b', p', rfl⟩ := kind_fwait kind
    have := ((acc g).poll hx).1
    rw [(nth_spec hp).2] at this
    exact this
  · show (e.track g).at (nth .bfence (prog g) b) ≺ (e.track g).at (some q)
    exact M.ordered fun q' h => by cases h; exact nth_lt (by omega)
  · show (e.track g).at (some q) ≺ (e.track g).at (nth .bcommit (prog g) b)
    exact M.ordered fun q' h => ⟨q, rfl, lt_nth h hq (by simp [Instr.kind?]) (by omega)⟩

/-- T1: every read of an accepted program, at issue and while its batch is in flight, sees its own block. -/
theorem accepted_read_current_generation
    (acc : ∀ g, accepts d count (decide (g = lead)) (prog g) = true) (con : Consistent prog lead e)
    (hr : readAt prog e g q = some r) (issued : By r.issue cut)
    (active : r.issue = some cut ∨ ¬ By (e.retire r.group r.block) cut) :
    (writes (cta count prog lead e) count).Shows r.slot r.block r.block cut :=
  have acc' : AllAccepted d count prog lead := fun g => accepted (acc g)
  read_current_generation (fold_of_accepted acc' con) (acc' lead).depth
    (readCode_of_accepted acc' con hr) issued active

/-- T2: a fill of the read's slot for a later block starts its `expect_tx` only after the read's batch retired. -/
theorem accepted_overwrite_after_last_read
    (acc : ∀ g, accepts d count (decide (g = lead)) (prog g) = true) (con : Consistent prog lead e)
    (hr : readAt prog e g q = some r) (kb : k < count) (later : r.block < k)
    (same : (loadSlot (prog lead) k).getD 0 = r.slot) :
    e.retire r.group r.block ≺ (e.track lead).at (nth .expect (prog lead) k) :=
  have acc' : AllAccepted d count prog lead := fun g => accepted (acc g)
  overwrite_after_last_read (fold_of_accepted acc' con) (acc' lead).depth
    (readCode_of_accepted acc' con hr) kb later same

end

/-! ### Rejections -/

section
variable {d count q : Nat} {leader : Bool} {P : List Instr}

/-- A poll of block `i` with another parity than `(i / d) % 2`. -/
theorem wrong_parity_rejected (h : P[q]? = some (.mbarWait b p))
    (wrong : p ≠ counts (P.take q) .fwait / d % 2) : accepts d count leader P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).poll h).2.2.2.2.2.2

/-- A refill of block `j ≥ d - 1` issued before the CTA barrier of step `j - (d - 1)`. -/
theorem refill_without_barrier_rejected (h : P[q]? = some (.expectTx b n))
    (refill : d - 1 ≤ counts (P.take q) .expect)
    (early : counts (P.take q) .barrier < counts (P.take q) .expect - (d - 1) + 2) :
    accepts d count leader P = false :=
  Bool.eq_false_iff.mpr fun a => by
    have := ((accepted a).expect h).2.2.2.2 refill
    omega

/-- A WGMMA operand that names another slot than `block % d`. -/
theorem wrong_slot_rejected (h : P[q]? = some (.read b s)) (wrong : s ≠ b % d) :
    accepts d count leader P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).read h).2.2.2

/-- A TMA copy of fill `j` into another slot than `j % d`. -/
theorem wrong_fill_slot_rejected (h : P[q]? = some (.tmaLoad sl b src n))
    (wrong : sl ≠ (counts (P.take q) .expect - 1) % d) : accepts d count leader P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).load h).2.1

/-- A step's `wgmma_wait` that keeps more than the newest group in flight, or fewer. -/
theorem wrong_wait_rejected (h : P[q]? = some (.wgmmaWait n)) (step : counts (P.take q) .bwait < count)
    (wrong : n ≠ 1) : accepts d count leader P = false :=
  Bool.eq_false_iff.mpr fun a => by
    have := (accepted a).bwait h
    simp only [step, ite_true] at this
    exact wrong this.1

end

/-! ### Diagnostics -/

def diagnoseRole (d count : Nat) (leader : Bool) (P : List Instr) : Option (Nat × String) :=
  if ¬ (2 ≤ d ∧ d - 1 ≤ count) then some (0, "the cooperative model needs 2 ≤ d and d - 1 ≤ count")
  else match diagnose (rules d count leader) P with
    | some r => some r
    | none => if bytesMatch P then none else some (P.length, "an expect_tx differs from the bytes of its tma_loads")

/-- The executable's verdict `accepted` is the verified checker's. -/
theorem diagnoseRole_none (h : diagnoseRole d count leader P = none) : accepts d count leader P = true := by
  unfold diagnoseRole at h
  split at h
  · cases h
  · rename_i pre
    split at h
    · cases h
    · rename_i hd
      split at h
      · rename_i hb
        simp only [accepts, Bool.and_eq_true, decide_eq_true_eq]
        exact ⟨⟨Classical.not_not.mp pre, diagnose_none hd⟩, hb⟩
      · cases h

example : accepts 2 3 true (expected 2 3 true 2 64 2) = true := by decide
example : accepts 3 5 false (expected 3 5 false 1 128 1) = true := by decide

/-- The refill of block 2 precedes the CTA barrier of step 1. -/
example : accepts 2 3 true
    [.init (.full 0) 1, .init (.full 1) 1, .barrier,
     .expectTx (.full 0) 8, .tmaLoad 0 (.full 0) 0 8,
     .mbarWait (.full 0) 0, .wgmmaFence, .read 0 0, .wgmmaCommit, .wgmmaWait 1, .barrier,
     .expectTx (.full 1) 8, .tmaLoad 1 (.full 1) 1 8,
     .mbarWait (.full 1) 0, .wgmmaFence, .read 1 1, .wgmmaCommit, .wgmmaWait 1,
     .expectTx (.full 0) 8, .tmaLoad 0 (.full 0) 2 8, .barrier,
     .mbarWait (.full 0) 1, .wgmmaFence, .read 2 0, .wgmmaCommit, .wgmmaWait 1, .barrier,
     .wgmmaWait 0, .settle] = false := by decide

/-- The same program with the refill after the barrier. -/
example : accepts 2 3 true
    [.init (.full 0) 1, .init (.full 1) 1, .barrier,
     .expectTx (.full 0) 8, .tmaLoad 0 (.full 0) 0 8,
     .mbarWait (.full 0) 0, .wgmmaFence, .read 0 0, .wgmmaCommit, .wgmmaWait 1, .barrier,
     .expectTx (.full 1) 8, .tmaLoad 1 (.full 1) 1 8,
     .mbarWait (.full 1) 0, .wgmmaFence, .read 1 1, .wgmmaCommit, .wgmmaWait 1, .barrier,
     .expectTx (.full 0) 8, .tmaLoad 0 (.full 0) 2 8,
     .mbarWait (.full 0) 1, .wgmmaFence, .read 2 0, .wgmmaCommit, .wgmmaWait 1, .barrier,
     .wgmmaWait 0, .settle] = true := by decide

end Ring.Cooperative
