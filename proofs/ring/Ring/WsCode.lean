import Ring.WarpSpecialized
import Ring.TmaCode

set_option linter.unusedSimpArgs false

namespace Ring.WarpSpecialized
open Timeline EmptyBarrier Tma

/-- The producer warp's first thread, or the leader lane of a consumer warp; the `initiator` sets the barriers up. -/
inductive Role where
  | producer
  | consumer (initiator : Bool)
  deriving DecidableEq, Repr

/-- Local rules of the warp-specialized fold (`lag = 1`, `nth i = i`, two consumer warpgroups). -/
def rules (d count : Nat) : Role → Rules
  | .producer => fun s prev x => match x with
    | .barrier => [("more than the set-up barrier", decide (s .barrier = 0))]
    | .mbarWait b p =>
      [("more empty waits than blocks", decide (s .ewait < count)),
       ("empty wait out of place", decide (s .barrier = 1 ∧ s .expect = s .ewait)),
       ("mbar_wait barrier is not empty J % d", decide (b = .empty (s .ewait % d))),
       ("mbar_wait parity is not (J / d + 1) % 2", decide (p = (s .ewait / d + 1) % 2))]
    | .expectTx b _ =>
      [("fill without its empty wait", decide (s .ewait = s .expect + 1)),
       ("expect_tx barrier is not full J % d", decide (b = .full (s .expect % d)))]
    | .tmaLoad sl b src _ =>
      [("tma_load outside a fill", inFill prev && decide (0 < s .expect)),
       ("tma_load slot is not J % d", decide (sl = (s .expect - 1) % d)),
       ("tma_load barrier is not full J % d", decide (b = .full ((s .expect - 1) % d))),
       ("tma_load source is not block J", decide (src = s .expect - 1))]
    | .wgmmaWait n => [("producer wgmma_wait out of place", decide (n = 0 ∧ s .bwait = 0 ∧ s .expect = count))]
    | .settle => [("settle out of place", decide (s .bwait = 1 ∧ s .expect = count))]
    | _ => [("not a producer instruction", false)]
  | .consumer initiator => fun s _ x => match x with
    | .init b n =>
      [("init outside the initiator's set-up", initiator && s .barrier == 0),
       ("init is not full I 1 then empty I 8, in order", decide (if s .init < d then b = .full (s .init) ∧ n = 1
         else s .init < 2 * d ∧ b = .empty (s .init - d) ∧ n = 8))]
    | .barrier =>
      [("more than the set-up barrier", decide (s .barrier = 0)),
       ("set-up barrier before all barriers are initialized", decide (initiator = true → s .init = 2 * d))]
    | .mbarWait b p =>
      [("more full waits than blocks", decide (s .fwait < count)),
       ("full wait before the previous step's wgmma_wait and release",
         decide (s .barrier = 1 ∧ s .bfence = s .fwait ∧ s .bwait = s .fwait ∧ s .arrive = s .fwait - 1)),
       ("mbar_wait barrier is not full I % d", decide (b = .full (s .fwait % d))),
       ("mbar_wait parity is not (I / d) % 2", decide (p = s .fwait / d % 2))]
    | .wgmmaFence =>
      [("wgmma_fence out of place",
        decide (s .bfence + 1 = s .fwait ∧ s .bcommit = s .bfence ∧ s .bwait = s .bfence))]
    | .read b sl =>
      [("read of another block than the step's", decide (b + 1 = s .fwait)),
       ("read outside the step's open batch", decide (s .bfence = s .fwait ∧ s .bcommit + 1 = s .bfence)),
       ("read slot is not block % d", decide (sl = b % d))]
    | .wgmmaCommit => [("wgmma_commit without an open batch", decide (s .bcommit + 1 = s .bfence))]
    | .wgmmaWait n =>
      [("wgmma_wait out of place", decide (if s .bwait < count then
          n = 1 ∧ s .bcommit = s .bwait + 1 ∧ s .arrive = s .bwait - 1
        else s .bwait = count ∧ n = 0 ∧ s .bcommit = count ∧ s .arrive = count - 1))]
    | .mbarArrive b =>
      [("more releases than blocks", decide (s .arrive < count)),
       ("release before the wgmma_wait that retires its block", decide (s .bwait = s .arrive + 2)),
       ("mbar_arrive barrier is not empty K % d", decide (b = .empty (s .arrive % d)))]
    | .settle =>
      [("settle out of place", decide (s .bwait = count + 1 ∧ s .arrive = count ∧ s .fwait = count))]
    | _ => [("not a consumer instruction", false)]

def accepts (d count : Nat) (role : Role) (P : List Instr) : Bool :=
  decide (0 < d) && run (rules d count role) P && bytesMatch P

/-! ### The canonical sequences -/

def fillOps (d loads bytes j : Nat) : List Instr :=
  .expectTx (.full (j % d)) (loads * bytes) :: List.replicate loads (.tmaLoad (j % d) (.full (j % d)) j bytes)

def produceOps (d loads bytes : Nat) : Nat → List Instr
  | 0 => []
  | j + 1 => produceOps d loads bytes j ++ .mbarWait (.empty (j % d)) ((j / d + 1) % 2) :: fillOps d loads bytes j

def releaseOps (d i : Nat) : List Instr := if 0 < i then [.mbarArrive (.empty ((i - 1) % d))] else []

def stepOps (d mmas i : Nat) : List Instr :=
  .mbarWait (.full (i % d)) (i / d % 2) :: .wgmmaFence :: (List.replicate mmas (.read i (i % d)) ++
  .wgmmaCommit :: .wgmmaWait 1 :: releaseOps d i)

def consumeOps (d mmas : Nat) : Nat → List Instr
  | 0 => []
  | i + 1 => consumeOps d mmas i ++ stepOps d mmas i

def initOps (d : Nat) : Nat → List Instr
  | 0 => []
  | n + 1 => initOps d n ++ [if n < d then .init (.full n) 1 else .init (.empty (n - d)) 8]

def expected (d count : Nat) (role : Role) (loads bytes mmas : Nat) : List Instr :=
  match role with
  | .producer => .barrier :: produceOps d loads bytes count ++ [.wgmmaWait 0, .settle]
  | .consumer initiator =>
    (if initiator then initOps d (2 * d) else []) ++ .barrier :: consumeOps d mmas count ++
    .wgmmaWait 0 :: releaseOps d count ++ [.settle]

private def pstate (j : Nat) : Counts
  | .barrier => 1
  | .ewait | .expect => j
  | _ => 0

private def cstate (initiator : Bool) (d i : Nat) : Counts
  | .init => if initiator then 2 * d else 0
  | .barrier => 1
  | .fwait | .bfence | .bcommit | .bwait => i
  | .arrive => i - 1
  | _ => 0

private def inits (n : Nat) : Counts
  | .init => n
  | _ => 0

section
variable {d count : Nat}

private theorem steps_produce (hl : 0 < loads) (hj : j ≤ count) :
    Steps (rules d count .producer) (pstate 0) (pstate j) (produceOps d loads bytes j) := by
  induction j with
  | zero => exact Steps.nil
  | succ j ih =>
    refine (ih (by omega)).append (Steps.cons ?_ (by simp [pstate]) rfl
      ((Steps.fill ?_ ?_ (by simp [pstate, Counts.add, Instr.kind?]) hl).eq ?_))
    · have : j < count := by omega
      intro p _; simp [rules, pstate, this]
    · intro p _; simp [rules, pstate, Counts.add, Instr.kind?]
    · intro p hp; simp [rules, pstate, Counts.add, Instr.kind?, hp]
    · funext k; cases k <;> simp [pstate, Counts.add, Instr.kind?]

private theorem steps_inits (hn : n ≤ 2 * d) :
    Steps (rules d count (.consumer true)) (counts []) (inits n) (initOps d n) := by
  induction n with
  | zero => exact Steps.nil.eq (by funext k; cases k <;> rfl)
  | succ n ih =>
    refine (ih (by omega)).append ((Steps.single ?_ rfl ?_).eq ?_)
    · intro p _
      by_cases h : n < d
      · simp [rules, inits, h]
      · simp [rules, inits, h]; omega
    · split <;> rfl
    · funext k; cases k <;> (split <;> simp [inits, Counts.add, Instr.kind?])

private theorem steps_step (hi : i < count) :
    Steps (rules d count (.consumer initiator)) (cstate initiator d i) (cstate initiator d (i + 1))
      (stepOps d mmas i) := by
  unfold stepOps
  refine Steps.cons ?_ (by simp [cstate]) rfl (Steps.cons ?_ (by simp [cstate, Counts.add, Instr.kind?]) rfl
    ((Steps.replicate ?_ (by simp [cstate, Counts.add, Instr.kind?]) rfl rfl).append
    (Steps.cons ?_ (by simp [cstate, Counts.add, Instr.kind?]) rfl
    (Steps.cons ?_ (by simp [cstate, Counts.add, Instr.kind?]) rfl ?_))))
  · intro p _; simp [rules, cstate, hi]
  · intro p _; simp [rules, cstate, Counts.add, Instr.kind?]
  · intro p _; simp [rules, cstate, Counts.add, Instr.kind?]
  · intro p _; simp [rules, cstate, Counts.add, Instr.kind?]
  · intro p _; simp [rules, cstate, Counts.add, Instr.kind?, hi]
  · unfold releaseOps
    split
    · refine (Steps.single ?_ (by simp [cstate, Counts.add, Instr.kind?]) rfl).eq ?_
      · intro p _; simp [rules, cstate, Counts.add, Instr.kind?]; omega
      · funext k; cases k <;> simp [cstate, Counts.add, Instr.kind?] <;> omega
    · refine Steps.nil.eq ?_
      funext k; cases k <;> simp [cstate, Counts.add, Instr.kind?] <;> omega

private theorem plain (h : x.kind? ≠ some .expect) (hl : isLoad x = false) : Balanced [x] := Balanced.plain h hl

private theorem cons_plain (h : x.kind? ≠ some .expect) (hl : isLoad x = false)
    (b : Balanced L) : Balanced (x :: L) := by
  rw [← List.singleton_append]; exact (Balanced.plain h hl).append b

private theorem balanced_release : Balanced (releaseOps d i) := by
  unfold releaseOps; split
  · exact plain (by simp [Instr.kind?]) rfl
  · exact Balanced.nil

private theorem balanced_expected : Balanced (expected d count role loads bytes mmas) := by
  cases role with
  | producer =>
    have produce (j : Nat) : Balanced (produceOps d loads bytes j) := by
      induction j with
      | zero => exact Balanced.nil
      | succ j ih => exact ih.append (cons_plain (by simp [Instr.kind?]) rfl Balanced.fill)
    exact (cons_plain (by simp [Instr.kind?]) rfl (produce _)).append
      (cons_plain (by simp [Instr.kind?]) rfl (plain (by simp [Instr.kind?]) rfl))
  | consumer initiator =>
    have inits (n : Nat) : Balanced (initOps d n) := by
      induction n with
      | zero => exact Balanced.nil
      | succ n ih => exact ih.append (plain (by split <;> simp [Instr.kind?]) (by split <;> rfl))
    have step (i : Nat) : Balanced (stepOps d mmas i) := by
      unfold stepOps
      exact cons_plain (by simp [Instr.kind?]) rfl (cons_plain (by simp [Instr.kind?]) rfl
        ((Balanced.replicate (plain (by simp [Instr.kind?]) rfl)).append
        (cons_plain (by simp [Instr.kind?]) rfl (cons_plain (by simp [Instr.kind?]) rfl balanced_release))))
    have consume (n : Nat) : Balanced (consumeOps d mmas n) := by
      induction n with
      | zero => exact Balanced.nil
      | succ n ih => exact ih.append (step n)
    unfold expected
    refine ((Balanced.append ?_ (cons_plain (by simp [Instr.kind?]) rfl (consume _))).append
      (cons_plain (by simp [Instr.kind?]) rfl balanced_release)).append (plain (by simp [Instr.kind?]) rfl)
    split
    · exact inits _
    · exact Balanced.nil

theorem expected_accepted (hd : 0 < d) (hl : 0 < loads) :
    accepts d count role (expected d count role loads bytes mmas) = true := by
  unfold accepts
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨hd, ?_⟩, bytesMatch_of balanced_expected⟩
  cases role with
  | producer =>
    have bar : Steps (rules d count .producer) (counts []) (pstate 0) [.barrier] := by
      refine (Steps.single ?_ rfl rfl).eq ?_
      · intro p _; simp [rules]
      · funext k; cases k <;> simp [pstate, Counts.add, Instr.kind?]
    have tail : Steps (rules d count .producer) (pstate count)
        ((pstate count).add (.wgmmaWait 0) |>.add .settle) [.wgmmaWait 0, .settle] := by
      refine Steps.cons ?_ (by simp [pstate]) rfl (Steps.single ?_ (by simp [pstate, Counts.add, Instr.kind?]) rfl)
      · intro p _; simp [rules, pstate]
      · intro p _; simp [rules, pstate, Counts.add, Instr.kind?]
    have all := (bar.append (steps_produce (bytes := bytes) hl (Nat.le_refl count))).append tail
    exact run_of_steps (by simpa [expected] using all) (by simp [pstate, Counts.add, Instr.kind?])
  | consumer initiator =>
    have setup : Steps (rules d count (.consumer initiator)) (counts []) (cstate initiator d 0)
        ((if initiator then initOps d (2 * d) else []) ++ [.barrier]) := by
      cases initiator with
      | false =>
        refine (Steps.single ?_ rfl rfl).eq ?_
        · intro p _; simp [rules]
        · funext k; cases k <;> simp [cstate, Counts.add, Instr.kind?]
      | true =>
        refine (steps_inits (Nat.le_refl _)).append ((Steps.single ?_ rfl rfl).eq ?_)
        · intro p _; simp [rules, inits]
        · funext k; cases k <;> simp [cstate, inits, Counts.add, Instr.kind?]
    have steps (n : Nat) (hn : n ≤ count) :
        Steps (rules d count (.consumer initiator)) (cstate initiator d 0) (cstate initiator d n)
          (consumeOps d mmas n) := by
      induction n with
      | zero => exact Steps.nil
      | succ n ih => exact (ih (by omega)).append (steps_step (by omega))
    have wait : Steps (rules d count (.consumer initiator)) (cstate initiator d count)
        ((cstate initiator d count).add (.wgmmaWait 0)) [.wgmmaWait 0] :=
      Steps.single (by intro p _; simp [rules, cstate]) (by simp [cstate]) rfl
    let fin : Counts := fun k => match k with
      | .init => if initiator then 2 * d else 0
      | .barrier => 1
      | .fwait | .bfence | .bcommit => count
      | .bwait => count + 1
      | .arrive => count
      | _ => 0
    have release : Steps (rules d count (.consumer initiator)) ((cstate initiator d count).add (.wgmmaWait 0))
        fin (releaseOps d count) := by
      unfold releaseOps
      split
      · refine (Steps.single ?_ (by simp [cstate, Counts.add, Instr.kind?]) rfl).eq ?_
        · intro p _; simp [rules, cstate, Counts.add, Instr.kind?]; omega
        · funext k; cases k <;> simp [fin, cstate, Counts.add, Instr.kind?] <;> omega
      · refine Steps.nil.eq ?_
        funext k; cases k <;> simp [fin, cstate, Counts.add, Instr.kind?] <;> omega
    have last : Steps (rules d count (.consumer initiator)) fin (fin.add .settle) [.settle] :=
      Steps.single (by intro p _; simp [rules, fin]) (by simp [fin]) rfl
    have all := ((setup.append (steps count (Nat.le_refl _))).append wait).append (release.append last)
    exact run_of_steps (by simpa [expected] using all) (by simp [fin, Counts.add, Instr.kind?])

end

/-! ### Rules at a position -/

structure Accepted (d count : Nat) (role : Role) (P : List Instr) : Prop where
  depth : 0 < d
  obeys : Obeys (rules d count role) P

theorem accepted (h : accepts d count role P = true) : Accepted d count role P := by
  unfold accepts at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, obeys_of_run h.1.2⟩

section
variable {d count : Nat} {P : List Instr}

section
variable (acc : Accepted d count .producer P)
include acc

theorem Accepted.ewait (h : P[q]? = some (.mbarWait b p)) :
    counts (P.take q) .ewait < count ∧ counts (P.take q) .expect = counts (P.take q) .ewait ∧
    b = .empty (counts (P.take q) .ewait % d) ∧ p = (counts (P.take q) .ewait / d + 1) % 2 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.1, r.2.1.2, r.2.2⟩

theorem Accepted.expect (h : P[q]? = some (.expectTx b n)) :
    counts (P.take q) .ewait = counts (P.take q) .expect + 1 ∧ b = .full (counts (P.take q) .expect % d) := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r

theorem Accepted.load (h : P[q]? = some (.tmaLoad sl b src n)) :
    0 < counts (P.take q) .expect ∧ sl = (counts (P.take q) .expect - 1) % d ∧
    b = .full ((counts (P.take q) .expect - 1) % d) ∧ src = counts (P.take q) .expect - 1 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.1.2, r.2⟩

theorem Accepted.produced : counts P .expect = count := by
  obtain ⟨qs, hs, hx, -⟩ := acc.obeys.settleAt
  have r := acc.obeys.rule qs _ hx
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  rw [acc.obeys.total hs]; simp [r.2]

theorem Accepted.fill (hj : j < count) :
    ∃ q n m, nth .expect P j = some q ∧ P[q]? = some (.expectTx (.full (j % d)) n) ∧
      P[q + 1]? = some (.tmaLoad (j % d) (.full (j % d)) j m) := by
  obtain ⟨q, hq, -⟩ := nth_lt (k := .expect) (n := j) (p := P.length) (P := P)
    (by rw [List.take_length, acc.produced]; exact hj)
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
  exact ⟨q, n, m, hq, ex.2 ▸ hx, hy⟩

end

section
variable (acc : Accepted d count (.consumer initiator) P)
include acc

theorem Accepted.poll (h : P[q]? = some (.mbarWait b p)) :
    counts (P.take q) .fwait < count ∧ counts (P.take q) .bfence = counts (P.take q) .fwait ∧
    counts (P.take q) .bwait = counts (P.take q) .fwait ∧ counts (P.take q) .arrive = counts (P.take q) .fwait - 1 ∧
    b = .full (counts (P.take q) .fwait % d) ∧ p = counts (P.take q) .fwait / d % 2 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨r.1, r.2.1.2.1, r.2.1.2.2.1, r.2.1.2.2.2, r.2.2⟩

theorem Accepted.bfence (h : P[q]? = some .wgmmaFence) :
    counts (P.take q) .bfence + 1 = counts (P.take q) .fwait := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r.1

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
      counts (P.take q) .arrive = counts (P.take q) .bwait - 1
    else counts (P.take q) .bwait = count ∧ n = 0 ∧ counts (P.take q) .bcommit = count ∧
      counts (P.take q) .arrive = count - 1 := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r

theorem Accepted.arrive (h : P[q]? = some (.mbarArrive b)) :
    counts (P.take q) .arrive < count ∧ counts (P.take q) .bwait = counts (P.take q) .arrive + 2 ∧
    b = .empty (counts (P.take q) .arrive % d) := by
  have r := acc.obeys.rule q _ h
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact r

theorem Accepted.final : ∃ qs, nth .settle P 0 = some qs ∧ counts (P.take qs) .bwait = count + 1 ∧
    counts (P.take qs) .arrive = count ∧ counts (P.take qs) .fwait = count := by
  obtain ⟨qs, hs, hx, -⟩ := acc.obeys.settleAt
  have r := acc.obeys.rule qs _ hx
  simp only [rules, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at r
  exact ⟨qs, hs, r⟩

theorem Accepted.released : counts P .arrive = count := by
  obtain ⟨qs, hs, -, a, -⟩ := acc.final
  rw [acc.obeys.total hs]; simp [a]

theorem Accepted.polled (hi : i < count) :
    ∃ q p, nth .fwait P i = some q ∧ P[q]? = some (.mbarWait (.full (i % d)) p) ∧ p = i / d % 2 ∧
      counts (P.take q) .arrive = i - 1 := by
  obtain ⟨qs, hs, -, -, f⟩ := acc.final
  obtain ⟨q, hq, -⟩ := nth_lt (k := .fwait) (n := i) (p := qs) (P := P) (by rw [f]; exact hi)
  obtain ⟨x, hx, kind⟩ := nth_instr hq
  obtain ⟨b, p, rfl⟩ := kind_fwait kind
  have r := acc.poll hx
  rw [(nth_spec hq).2] at r
  obtain ⟨-, -, -, a, hb, hp⟩ := r
  cases hb
  exact ⟨q, p, hq, hx, hp, a⟩

theorem Accepted.gave (hj : j < count) :
    ∃ q, nth .arrive P j = some q ∧ P[q]? = some (.mbarArrive (.empty (j % d))) ∧
      counts (P.take q) .bwait = j + 2 := by
  obtain ⟨qs, hs, -, a, -⟩ := acc.final
  obtain ⟨q, hq, -⟩ := nth_lt (k := .arrive) (n := j) (p := qs) (P := P) (by rw [a]; exact hj)
  obtain ⟨x, hx, kind⟩ := nth_instr hq
  obtain ⟨b, rfl⟩ := kind_arrive kind
  have r := acc.arrive hx
  rw [(nth_spec hq).2] at r
  obtain ⟨-, w, rfl⟩ := r
  exact ⟨q, hq, hx, w⟩

theorem Accepted.waitAt (h : nth .bwait P i = some q) :
    ∃ n, P[q]? = some (.wgmmaWait n) ∧ (if i < count then n = 1 ∧ counts (P.take q) .bcommit = i + 1
      else i = count ∧ n = 0 ∧ counts (P.take q) .bcommit = count) := by
  obtain ⟨x, hx, kind⟩ := nth_instr h
  obtain ⟨n, rfl⟩ := kind_bwait kind
  have r := acc.bwait hx
  rw [(nth_spec h).2] at r
  refine ⟨n, hx, ?_⟩
  split <;> simp_all

end
end

/-! ### Physical slots -/

theorem tally_congr {p q : Nat → Bool} (h : ∀ k, k < n → p k = q k) : tally p n = tally q n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [tally, ih (fun k hk => h k (by omega)), h n (by omega)]

private theorem eq_of_lt_iff {a b : Nat} (h : ∀ x, x < a ↔ x < b) : a = b := by
  have := h a; have := h b; omega

private theorem turns_mod (hd : 0 < d) (hn : n % d = s) : turns d n s = n / d := by
  apply eq_of_lt_iff
  intro h
  rw [inTurns hd]
  have := Nat.mod_add_div n d
  constructor
  · intro lt
    apply Classical.byContradiction
    intro ge
    have := Nat.mul_le_mul_left d (Nat.le_of_not_lt ge)
    omega
  · intro lt
    have := Nat.mul_le_mul_left d (Nat.succ_le_of_lt lt)
    rw [Nat.mul_succ] at this
    omega

private theorem turns_succ (hd : 0 < d) (hs : s < d) :
    turns d (n + 1) s = turns d n s + if n % d = s then 1 else 0 := by
  have dm := Nat.mod_add_div n d
  split
  · rename_i hn
    rw [turns_mod hd hn]
    apply eq_of_lt_iff
    intro h
    rw [inTurns hd]
    constructor
    · intro lt
      apply Classical.byContradiction
      intro ge
      have := Nat.mul_le_mul_left d (by omega : n / d + 1 ≤ h)
      rw [Nat.mul_succ] at this
      omega
    · intro lt
      have := Nat.mul_le_mul_left d (by omega : h ≤ n / d)
      omega
  · rename_i hn
    apply eq_of_lt_iff
    intro h
    show h < turns d (n + 1) s ↔ h < turns d n s
    rw [inTurns hd, inTurns hd]
    constructor
    · intro lt
      rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ lt) with lt | eq
      · exact lt
      · exfalso; apply hn
        rw [← eq, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hs]
    · intro lt; omega

/-- Counting the fills of one physical slot by block index or by visit gives the same number. -/
theorem tally_slot (hd : 0 < d) (hs : s < d) (f : Nat → Bool) (n : Nat) :
    tally (fun j => decide (j % d = s) && f j) n = tally (fun h => f (s + d * h)) (turns d n s) := by
  induction n with
  | zero => simp [tally, turns]
  | succ n ih =>
    rw [turns_succ hd hs]
    simp only [tally, ih]
    by_cases hn : n % d = s
    · have at_ : s + d * turns d n s = n := by
        rw [turns_mod hd hn]; have := Nat.mod_add_div n d; omega
      simp [tally, at_, hn]
    · simp [hn]

/-! ### Timelines consistent with program order -/

/-- The producer's and both consumer groups' timelines, the device completion of each fill and of each
    WGMMA batch, and the `k`-th arrival of the leader lane of each of the eight consumer warps. -/
structure Exec where
  producer : Track
  consumer : Fin 2 → Track
  arrived : Nat → Event
  landed : Nat → Event
  publish : Nat → Event
  retire : Fin 2 → Nat → Event
  give : Fin 8 → Nat → Event

/-- The empty barrier of the `k`-th release of a consumer program. -/
def arriveBar (P : List Instr) (k : Nat) : Nat := (barOf P .arrive k).getD 0

/-- Program order and the mbarrier and WGMMA hardware rules.
    A full barrier's completed phase is one published fill; an empty barrier's phase completes per eight arrivals.
    A consumer warpgroup runs its program converged; each warp leader arrives between the neighbours of its `mbar_arrive`. -/
structure Consistent (prod : List Instr) (cons : Fin 2 → List Instr) (e : Exec) : Prop where
  produce : e.producer.Mono
  consume : ∀ g, (e.consumer g).Mono
  expectArrive : ∀ j, e.producer.at (nth .expect prod j) ≺ e.arrived j
  loadLanded : ∀ q sl b src n, prod[q]? = some (.tmaLoad sl b src n) →
    e.producer.at (some q) ≺ e.landed (counts (prod.take q) .expect - 1)
  noFill : ∀ j, nth .expect prod j = none → e.landed j = none
  arrivedPublish : ∀ j, e.arrived j ≺ e.publish j
  landedPublish : ∀ j, e.landed j ≺ e.publish j
  fullParity : ∀ g q b p t, (cons g)[q]? = some (.mbarWait (.full b) p) → (e.consumer g).at (some q) = some t →
    phaseAt (fillBar prod) e.publish (counts prod .expect) b t % 2 ≠ p
  emptyParity : ∀ q b p t, prod[q]? = some (.mbarWait (.empty b) p) → e.producer.at (some q) = some t →
    total 8 (fun w => tally (fun k => decide (arriveBar (cons (group w)) k = b) && occurred (e.give w k) t)
      (counts (cons (group w)) .arrive)) / 8 % 2 ≠ p
  giveAfter : ∀ w k q, nth .arrive (cons (group w)) k = some q → ∀ p, p < q →
    (e.consumer (group w)).at (some p) ≺ e.give w k
  giveBefore : ∀ w k q, nth .arrive (cons (group w)) k = some q →
    e.give w k ≺ (e.consumer (group w)).at (some (q + 1))
  retire : ∀ g c, (e.consumer g).at (nth .bcommit (cons g) c) ≺ e.retire g c
  waitGroup : ∀ g q n c, (cons g)[q]? = some (.wgmmaWait n) →
    (e.consumer g).at (nth .bcommit (cons g) (c + n)) ≺ (e.consumer g).at (some q) →
    e.retire g c ≺ (e.consumer g).at (some q)

/-- The model CTA read off the programs and the timeline. Visit `h` of slot `s` is block `s + d * h`;
    the slot operands decide which events the model's per-slot projections see. -/
def cta (d count : Nat) (prod : List Instr) (cons : Fin 2 → List Instr) (e : Exec) : CTA where
  slots s := {
    acquire := fun h => e.producer.at (nth .ewait prod (s + d * h))
    start := fun h => e.producer.at (nth .expect prod (s + d * h))
    publish := fun h => e.publish (s + d * h)
    poll := fun g h => (e.consumer g).at (nth .fwait (cons g) (s + d * h))
    give := fun w h => e.give w (s + d * h) }
  emptySlot j := (barOf prod .ewait j).getD (j % d)
  panel j := (loadSlot prod j).getD (j % d)
  fullSlot j := (barOf prod .expect j).getD (j % d)
  pollSlot g j := (barOf (cons g) .fwait j).getD (j % d)
  giveSlot w j := (barOf (cons (group w)) .arrive j).getD (j % d)
  source j := (loadSrc prod j).getD j
  arrived := e.arrived
  bytesComplete := e.landed
  batchFence g j := (e.consumer g).at (nth .bfence (cons g) j)
  commit g j := (e.consumer g).at (nth .bcommit (cons g) j)
  done_ := e.retire
  waitGroup g j := (e.consumer g).at (if j < count then nth .bwait (cons g) j else none)
  finalWait g := (e.consumer g).at (nth .bwait (cons g) count)

def readAt (cons : Fin 2 → List Instr) (e : Exec) (g : Fin 2) (q : Nat) : Option Read :=
  match (cons g)[q]? with
  | some (.read b s) => some ⟨g, b, s, (e.consumer g).at (some q)⟩
  | _ => none

/-- The producer and both consumer programs pass; consumer 0 initializes the barriers. -/
structure AllAccepted (d count : Nat) (prod : List Instr) (cons : Fin 2 → List Instr) : Prop where
  producer : Accepted d count .producer prod
  consumer : ∀ g, Accepted d count (.consumer (g = 0)) (cons g)

section
variable {d count : Nat} {prod : List Instr} {cons : Fin 2 → List Instr} {e : Exec}

private theorem getD_mod {o : Option Nat} (h : ∀ x, o = some x → x = j % d) : o.getD (j % d) = j % d := by
  cases o with
  | none => rfl
  | some x => simp [h x rfl]

section
variable (acc : AllAccepted d count prod cons)
include acc

theorem AllAccepted.emptyOp (j : Nat) : (barOf prod .ewait j).getD (j % d) = j % d := getD_mod fun x h => by
  cases hq : nth .ewait prod j with
  | none => rw [barOf_none hq] at h; cases h
  | some q =>
    obtain ⟨y, hy, kind⟩ := nth_instr hq
    obtain ⟨b, p, rfl⟩ := kind_ewait kind
    have r := (acc.producer.ewait hy).2.2.1
    rw [(nth_spec hq).2] at r
    rw [barOf_wait hq hy, r] at h
    cases h; rfl

theorem AllAccepted.fullOp (j : Nat) : (barOf prod .expect j).getD (j % d) = j % d := getD_mod fun x h => by
  cases hq : nth .expect prod j with
  | none => rw [barOf_none hq] at h; cases h
  | some q =>
    obtain ⟨y, hy, kind⟩ := nth_instr hq
    obtain ⟨b, n, rfl⟩ := kind_expect kind
    have r := (acc.producer.expect hy).2
    rw [(nth_spec hq).2] at r
    rw [barOf_expectTx hq hy, r] at h
    cases h; rfl

theorem AllAccepted.panelOp (j : Nat) : (loadSlot prod j).getD (j % d) = j % d := getD_mod fun x h => by
  cases hq : nth .expect prod j with
  | none => simp [loadSlot, loadOf_none hq] at h
  | some q =>
    obtain ⟨y, hy, kind⟩ := nth_instr hq
    obtain ⟨b, n, rfl⟩ := kind_expect kind
    obtain ⟨sl, b', src, m, hz⟩ := acc.producer.obeys.next hy
    have r := (acc.producer.load hz).2.1
    rw [counts_at hy, (nth_spec hq).2] at r
    simp only [Instr.kind?, ite_true, Nat.add_sub_cancel] at r
    simp only [loadSlot, loadOf_some hq, hz] at h
    cases h; exact r

theorem AllAccepted.pollOp (g j) : (barOf (cons g) .fwait j).getD (j % d) = j % d := getD_mod fun x h => by
  cases hq : nth .fwait (cons g) j with
  | none => rw [barOf_none hq] at h; cases h
  | some q =>
    obtain ⟨y, hy, kind⟩ := nth_instr hq
    obtain ⟨b, p, rfl⟩ := kind_fwait kind
    have r := ((acc.consumer g).poll hy).2.2.2.2.1
    rw [(nth_spec hq).2] at r
    rw [barOf_wait hq hy, r] at h
    cases h; rfl

theorem AllAccepted.giveOp (g j) : (barOf (cons g) .arrive j).getD (j % d) = j % d := getD_mod fun x h => by
  cases hq : nth .arrive (cons g) j with
  | none => rw [barOf_none hq] at h; cases h
  | some q =>
    obtain ⟨y, hy, kind⟩ := nth_instr hq
    obtain ⟨b, rfl⟩ := kind_arrive kind
    have r := ((acc.consumer g).arrive hy).2.2
    rw [(nth_spec hq).2] at r
    rw [barOf_arrive hq hy, r] at h
    cases h; rfl

theorem AllAccepted.acquire (j) : (cta d count prod cons e).acquire d j = e.producer.at (nth .ewait prod j) := by
  simp only [CTA.acquire, cta, acc.emptyOp, Nat.mod_add_div]

theorem AllAccepted.start (j) : (cta d count prod cons e).start d j = e.producer.at (nth .expect prod j) := by
  simp only [CTA.start, cta, acc.panelOp, Nat.mod_add_div]

theorem AllAccepted.publish (j) : (cta d count prod cons e).publish d j = e.publish j := by
  simp only [CTA.publish, cta, acc.fullOp, Nat.mod_add_div]

theorem AllAccepted.poll (g j) : (cta d count prod cons e).poll d g j = (e.consumer g).at (nth .fwait (cons g) j) := by
  simp only [CTA.poll, cta, acc.pollOp, Nat.mod_add_div]

theorem AllAccepted.give (w j) : (cta d count prod cons e).give d w j = e.give w j := by
  simp only [CTA.give, cta, acc.giveOp, Nat.mod_add_div]

end

theorem fold_of_accepted (acc : AllAccepted d count prod cons) (con : Consistent prod cons e) :
    Fold (cta d count prod cons e) d count := by
  have A := acc.producer
  have C := acc.consumer
  have hd := A.depth
  have T := con.produce
  have U := con.consume
  have tail (g) (q : Nat) (h : nth .bwait (cons g) count = some q) :
      ∃ n, (cons g)[q]? = some (.wgmmaWait n) ∧ n = 0 ∧ counts ((cons g).take q) .bcommit = count := by
    obtain ⟨n, hx, r⟩ := (C g).waitAt h
    simp only [Nat.lt_irrefl, ite_false] at r
    exact ⟨n, hx, r.2.1, r.2.2⟩
  constructor
  · intro j _; exact acc.emptyOp j
  · intro j _; exact acc.panelOp j
  · intro j _; exact acc.fullOp j
  · intro g j _; exact acc.pollOp g j
  · intro w j _; exact acc.giveOp _ j
  · intro j hj
    obtain ⟨q, n, m, hq, -, hy⟩ := A.fill hj
    simp [cta, loadSrc, loadOf_some hq, hy]
  · intro j
    rw [acc.acquire, acc.start]
    exact T.ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, n, rfl⟩ := kind_expect kind
      have r := (A.expect hx).1
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro j _
    rw [acc.acquire, acc.start]
    exact T.ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, p, rfl⟩ := kind_ewait kind
      have r := (A.ewait hx).2.1
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro j; rw [acc.start]; exact con.expectArrive j
  · intro j
    rw [acc.start]
    cases h : nth .expect prod j with
    | none => simp only [cta, con.noFill j h, Track.at]; exact edge_none_right
    | some q =>
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, n, rfl⟩ := kind_expect kind
      obtain ⟨sl, b', src, m, hy⟩ := A.obeys.next hx
      have land := con.loadLanded _ _ _ _ _ hy
      rw [counts_at hx, (nth_spec h).2] at land
      simp only [Instr.kind?, ite_true, Nat.add_sub_cancel] at land
      exact (T.lt (Nat.lt_succ_self q)).trans land
  · intro j; rw [acc.publish]; exact con.arrivedPublish j
  · intro j; rw [acc.publish]; exact con.landedPublish j
  · intro g j t hj ht
    rw [acc.poll] at ht
    obtain ⟨q, p, hq, hx, hp, -⟩ := (C g).polled hj
    rw [hq] at ht
    have par := con.fullParity g q _ p t hx ht
    rw [A.produced] at par
    have eq : phaseAt (fillBar prod) e.publish count (j % d) t =
        fullPhase ((cta d count prod cons e).slots (j % d)) (turns d count (j % d)) t := by
      unfold phaseAt fullPhase
      rw [tally_congr (q := fun k => decide (k % d = j % d) && occurred (e.publish k) t), tally_slot hd (Nat.mod_lt _ hd)]
      · rfl
      · intro k hk
        obtain ⟨q', n, m, hq', hx', -⟩ := A.fill hk
        simp [fillBar, barOf_expectTx hq' hx', Bar.slot]
    have ps : (cta d count prod cons e).pollSlot g j = j % d := acc.pollOp g j
    show fullPhase ((cta d count prod cons e).slots ((cta d count prod cons e).pollSlot g j))
      (turns d count ((cta d count prod cons e).pollSlot g j)) t % 2 ≠ j / d % 2
    rw [ps, ← eq, ← hp]
    exact par
  · intro j t hj ht
    rw [acc.acquire] at ht
    obtain ⟨q', n, m, hq', hx', -⟩ := A.fill hj
    have hw := nth_lt (k := .ewait) (n := j) (P := prod) (p := q') (by
      have := (A.expect hx').1; rw [(nth_spec hq').2] at this; omega)
    obtain ⟨q, hq, -⟩ := hw
    obtain ⟨x, hx, kind⟩ := nth_instr hq
    obtain ⟨b, p, rfl⟩ := kind_ewait kind
    have r := A.ewait hx
    rw [(nth_spec hq).2] at r
    obtain ⟨-, -, hb, hp⟩ := r
    cases hb
    rw [hq] at ht
    have par := con.emptyParity q _ p t hx ht
    have eq : (total 8 fun w => tally (fun k => decide (arriveBar (cons (group w)) k = j % d) &&
        occurred (e.give w k) t) (counts (cons (group w)) .arrive)) / 8 =
        emptyPhase ((cta d count prod cons e).slots (j % d)) (turns d count (j % d)) t := by
      unfold emptyPhase credits
      congr 2
      funext w
      rw [(C (group w)).released,
        tally_congr (q := fun k => decide (k % d = j % d) && occurred (e.give w k) t), tally_slot hd (Nat.mod_lt _ hd)]
      · rfl
      · intro k hk
        obtain ⟨qa, ha, hxa, -⟩ := (C (group w)).gave hk
        simp [arriveBar, barOf_arrive ha hxa, Bar.slot]
    have es : (cta d count prod cons e).emptySlot j = j % d := acc.emptyOp j
    show emptyPhase ((cta d count prod cons e).slots ((cta d count prod cons e).emptySlot j))
      (turns d count ((cta d count prod cons e).emptySlot j)) t % 2 ≠ (j / d + 1) % 2
    rw [es, ← eq, ← hp]
    exact par
  · intro g j
    rw [acc.poll]
    exact (U g).ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      rw [kind_bfence kind] at hx
      have r := (C g).bfence hx
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro g j
    exact (U g).ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      rw [kind_bcommit kind] at hx
      have r := (C g).bcommit hx
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro g j; exact con.retire g j
  · intro g i
    simp only [cta]
    split
    · exact (U g).ordered fun q h => by
        obtain ⟨n, hx, r⟩ := (C g).waitAt h
        simp only [‹i < count›, ite_true] at r
        exact nth_lt (by omega)
    · exact edge_none_right
  · intro g j hj
    rw [acc.poll]
    simp only [cta, show j < count by omega, ite_true]
    exact (U g).ordered fun q h => by
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, p, rfl⟩ := kind_fwait kind
      have r := ((C g).poll hx).2.2.1
      rw [(nth_spec h).2] at r
      exact nth_lt (by omega)
  · intro g i j lt
    simp only [cta]
    split
    · rename_i hi
      cases h : nth .bwait (cons g) i with
      | none => exact edge_none_right
      | some q =>
        obtain ⟨n, hx, r⟩ := (C g).waitAt h
        simp only [hi, ite_true] at r
        obtain ⟨rfl, bc⟩ := r
        apply con.waitGroup g q 1 j hx
        exact (U g).ordered fun q' h' => by cases h'; exact nth_lt (by omega)
    · exact edge_none_right
  · intro g i hi
    cases h : nth .bwait (cons g) count with
    | none => simp only [cta, h]; exact edge_none_right
    | some q =>
      obtain ⟨n, hx, rfl, bc⟩ := tail g q h
      simp only [cta, h]
      apply con.waitGroup g q 0 i hx
      exact (U g).ordered fun q' h' => by cases h'; exact nth_lt (by omega)
  · intro w j hj
    rw [acc.give]
    obtain ⟨q, hq, -, bw⟩ := (C (group w)).gave (by omega : j < count)
    obtain ⟨p, hp, lt⟩ := nth_lt (k := .bwait) (n := j + 1) (P := cons (group w)) (p := q) (by omega)
    simp only [cta, hj, ite_true, hp]
    exact con.giveAfter w j q hq p lt
  · intro w j hj
    rw [acc.give]
    obtain ⟨q, hq, -, bw⟩ := (C (group w)).gave (by omega : j < count)
    obtain ⟨p, hp, lt⟩ := nth_lt (k := .bwait) (n := count) (P := cons (group w)) (p := q) (by omega)
    simp only [cta, hp]
    exact con.giveAfter w j q hq p lt
  · intro w j hj
    rw [acc.give, acc.poll]
    obtain ⟨q, hq, -, -⟩ := (C (group w)).gave (by omega : j < count)
    cases h : nth .fwait (cons (group w)) (j + 2) with
    | none => exact edge_none_right
    | some q' =>
      obtain ⟨x, hx, kind⟩ := nth_instr h
      obtain ⟨b, p, rfl⟩ := kind_fwait kind
      have r := ((C (group w)).poll hx).2.2.2.1
      rw [(nth_spec h).2] at r
      obtain ⟨q'', hq'', lt⟩ := nth_lt (k := .arrive) (n := j) (P := cons (group w)) (p := q') (by omega)
      rw [hq] at hq''; cases hq''
      exact (U (group w)).later (con.giveBefore w j q hq) (by omega)

end

section
variable {d count : Nat} {prod : List Instr} {cons : Fin 2 → List Instr} {e : Exec}

theorem readCode_of_accepted (acc : AllAccepted d count prod cons) (con : Consistent prod cons e)
    (hr : readAt cons e g q = some r) : ReadCode (cta d count prod cons e) d count r := by
  unfold readAt at hr
  split at hr
  case h_2 => cases hr
  rename_i b s hq
  cases hr
  have C := acc.consumer g
  have rr := C.read hq
  have M := con.consume g
  refine ⟨?_, rr.2.2.2, ?_, ?_⟩
  · obtain ⟨p, hp, lt⟩ := nth_lt (k := .fwait) (n := b) (P := cons g) (p := q) (by omega)
    obtain ⟨x, hx, kind⟩ := nth_instr hp
    obtain ⟨b', p', rfl⟩ := kind_fwait kind
    have := (C.poll hx).1
    rw [(nth_spec hp).2] at this
    exact this
  · show (e.consumer g).at (nth .bfence (cons g) b) ≺ (e.consumer g).at (some q)
    exact M.ordered fun q' h => by cases h; exact nth_lt (by omega)
  · show (e.consumer g).at (some q) ≺ (e.consumer g).at (nth .bcommit (cons g) b)
    exact M.ordered fun q' h => ⟨q, rfl, lt_nth h hq (by simp [Instr.kind?]) (by omega)⟩

/-- The producer and both consumer programs are accepted; consumer 0 initializes the barriers. -/
abbrev AllAccept (d count : Nat) (prod : List Instr) (cons : Fin 2 → List Instr) : Prop :=
  accepts d count .producer prod = true ∧ ∀ g, accepts d count (.consumer (g = 0)) (cons g) = true

theorem AllAccept.accepted (h : AllAccept d count prod cons) : AllAccepted d count prod cons :=
  ⟨WarpSpecialized.accepted h.1, fun g => WarpSpecialized.accepted (h.2 g)⟩

/-- T1: every WGMMA read of an accepted consumer, at issue and while its batch is in flight, sees its own block. -/
theorem accepted_read_current_generation (acc : AllAccept d count prod cons) (con : Consistent prod cons e)
    (hr : readAt cons e g q = some r) (issued : By r.issue cut)
    (active : r.issue = some cut ∨ ¬ By (e.retire r.group r.block) cut) :
    (writes (cta d count prod cons e) d count).Shows r.slot r.block r.block cut :=
  read_current_generation (fold_of_accepted acc.accepted con) acc.accepted.producer.depth
    (readCode_of_accepted acc.accepted con hr) issued active

/-- T2: the producer's fill of the read's slot for a later block starts only after the read's batch retired. -/
theorem accepted_overwrite_after_last_read (acc : AllAccept d count prod cons) (con : Consistent prod cons e)
    (hr : readAt cons e g q = some r) (kb : k < count) (later : r.block < k)
    (same : (loadSlot prod k).getD (k % d) = r.slot) :
    e.retire r.group r.block ≺ e.producer.at (nth .expect prod k) := by
  rw [← acc.accepted.start (e := e) k]
  exact overwrite_after_last_read (fold_of_accepted acc.accepted con) acc.accepted.producer.depth
    (readCode_of_accepted acc.accepted con hr) kb later same

end

/-! ### Rejections -/

section
variable {d count q : Nat} {P : List Instr}

/-- A consumer's poll of block `i` with another parity than `(i / d) % 2`. -/
theorem wrong_full_parity_rejected (h : P[q]? = some (.mbarWait b p))
    (wrong : p ≠ counts (P.take q) .fwait / d % 2) : accepts d count (.consumer i) P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).poll h).2.2.2.2.2

/-- The producer's wait for slot `j % d` with another parity than `(j / d + 1) % 2`. -/
theorem wrong_empty_parity_rejected (h : P[q]? = some (.mbarWait b p))
    (wrong : p ≠ (counts (P.take q) .ewait / d + 1) % 2) : accepts d count .producer P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).ewait h).2.2.2

/-- A consumer that gives fewer than `count` slots back. -/
theorem missing_release_rejected (few : counts P .arrive < count) : accepts d count (.consumer i) P = false :=
  Bool.eq_false_iff.mpr fun a => by have := (accepted a).released; omega

/-- A release of block `k` before the `wgmma_wait` that retires block `k`'s batch. -/
theorem early_release_rejected (h : P[q]? = some (.mbarArrive b))
    (early : counts (P.take q) .bwait < counts (P.take q) .arrive + 2) : accepts d count (.consumer i) P = false :=
  Bool.eq_false_iff.mpr fun a => by have := ((accepted a).arrive h).2.1; omega

/-- A WGMMA operand that names another slot than `block % d`. -/
theorem wrong_slot_rejected (h : P[q]? = some (.read b s)) (wrong : s ≠ b % d) :
    accepts d count (.consumer i) P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).read h).2.2.2

/-- A release on the empty barrier of another slot than `k % d`. -/
theorem wrong_release_slot_rejected (h : P[q]? = some (.mbarArrive b))
    (wrong : b ≠ .empty (counts (P.take q) .arrive % d)) : accepts d count (.consumer i) P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).arrive h).2.2

/-- A TMA copy of fill `j` into another slot than `j % d`. -/
theorem wrong_fill_slot_rejected (h : P[q]? = some (.tmaLoad sl b src n))
    (wrong : sl ≠ (counts (P.take q) .expect - 1) % d) : accepts d count .producer P = false :=
  Bool.eq_false_iff.mpr fun a => wrong ((accepted a).load h).2.1

end

/-! ### Diagnostics -/

def diagnoseRole (d count : Nat) (role : Role) (P : List Instr) : Option (Nat × String) :=
  if ¬ 0 < d then some (0, "the warp-specialized model needs 0 < d")
  else match diagnose (rules d count role) P with
    | some r => some r
    | none => if bytesMatch P then none else some (P.length, "an expect_tx differs from the bytes of its tma_loads")

/-- The executable's verdict `accepted` is the verified checker's. -/
theorem diagnoseRole_none (h : diagnoseRole d count role P = none) : accepts d count role P = true := by
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

example : accepts 4 9 .producer (expected 4 9 .producer 2 64 0) = true := by decide
example : accepts 4 9 (.consumer true) (expected 4 9 (.consumer true) 0 0 2) = true := by decide
example : accepts 2 0 (.consumer false) (expected 2 0 (.consumer false) 0 0 1) = true := by decide

/-- The release of block 0 precedes the `wgmma_wait 1` of step 1, which retires block 0's batch. -/
example : accepts 2 2 (.consumer false)
    [.barrier,
     .mbarWait (.full 0) 0, .wgmmaFence, .read 0 0, .wgmmaCommit, .wgmmaWait 1,
     .mbarWait (.full 1) 0, .wgmmaFence, .read 1 1, .wgmmaCommit, .mbarArrive (.empty 0), .wgmmaWait 1,
     .wgmmaWait 0, .mbarArrive (.empty 1), .settle] = false := by decide

/-- The same program with the release after the wait. -/
example : accepts 2 2 (.consumer false)
    [.barrier,
     .mbarWait (.full 0) 0, .wgmmaFence, .read 0 0, .wgmmaCommit, .wgmmaWait 1,
     .mbarWait (.full 1) 0, .wgmmaFence, .read 1 1, .wgmmaCommit, .wgmmaWait 1, .mbarArrive (.empty 0),
     .wgmmaWait 0, .mbarArrive (.empty 1), .settle] = true := by decide

end Ring.WarpSpecialized
