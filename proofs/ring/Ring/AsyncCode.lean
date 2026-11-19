import Ring.Async

namespace Ring.Async
open Timeline

deriving instance DecidableEq, Repr for Reader

/-- One thread's ring instructions in program order. A `fill` is one `cp.async`
    of ticket `ticket`; several may precede one `commit`. -/
inductive Instr where
  | fill (ticket slot src : Nat)
  | commit
  | waitGroup (pending : Nat)
  | proxyFence
  | barrier
  | read (block slot : Nat) (reader : Reader)
  | wgmmaFence (batch : Nat)
  | wgmmaCommit (batch : Nat)
  | wgmmaWait0 (batch : Nat)
  | settle
  deriving DecidableEq, Repr

/-- Instructions whose occurrences are numbered. -/
inductive Kind where
  | commit | wait | proxy | barrier | bfence | bcommit | bwait | settle
  deriving DecidableEq, Repr

def Instr.kind? : Instr → Option Kind
  | .commit => some .commit
  | .waitGroup _ => some .wait
  | .proxyFence => some .proxy
  | .barrier => some .barrier
  | .wgmmaFence _ => some .bfence
  | .wgmmaCommit _ => some .bcommit
  | .wgmmaWait0 _ => some .bwait
  | .settle => some .settle
  | _ => none

abbrev Counts := Kind → Nat

def counts (P : List Instr) : Counts := fun k => P.countP (·.kind? == some k)

def Counts.add (s : Counts) (x : Instr) : Counts := fun k => s k + if x.kind? = some k then 1 else 0

def isFill : Option Instr → Bool
  | some (.fill ..) => true
  | _ => false

/-- Local rules of the lag-zero cp.async fold with `exact=false`, evaluated
    against the counts of the preceding instructions. -/
def rules (d count : Nat) (proxy : Bool) (s : Counts) (prev : Option Instr) (x : Instr) :
    List (String × Bool) :=
  ("instruction after settle", s .settle == 0) :: match x with
  | .fill j slot src =>
    [("ticket is not the number of commits", j == s .commit),
     ("ticket beyond d-1+count", decide (j < d - 1 + count)),
     ("slot is not ticket % d", slot == j % d),
     ("source is not Async.source", src == source d count j),
     (if j < d - 1 then "prologue fill after a wait" else "refill without its barrier",
      if j < d - 1 then s .wait == 0 else s .wait == j - (d - 1) + 1 && s .barrier == s .wait)]
  | .commit => [("commit without a preceding fill", isFill prev)]
  | .waitGroup n =>
    [("wait immediate is not d-2", n == d - 2),
     ("more than count waits", decide (s .wait < count)),
     ("wait before the step's refill commit", s .commit == d - 1 + s .wait),
     ("wait before the step's barrier", s .barrier == s .wait),
     ("wait with an open wgmma batch", s .bfence == s .bwait)]
  | .proxyFence =>
    [("proxy fence without proxy", proxy),
     ("proxy fence not between wait and barrier",
       s .wait == s .proxy + 1 && s .barrier == s .proxy)]
  | .barrier =>
    [("barrier not after the step's wait", s .barrier + 1 == s .wait),
     ("barrier before the proxy fence", !proxy || s .proxy == s .wait)]
  | .read b slot rd =>
    [("read of another block than the step's", b + 1 == s .wait),
     ("read before the step's refill commit", s .commit == d - 1 + s .wait),
     ("read slot is not block % d", slot == b % d),
     ("wgmma read outside its open batch", match rd with
       | .load => true
       | .wgmma c => proxy && c + 1 == s .bfence && s .bcommit == c)]
  | .wgmmaFence c =>
    [("wgmma without proxy", proxy),
     ("wgmma batch number", c == s .bfence && s .bcommit == c && s .bwait == c),
     ("wgmma batch before the step's refill commit",
       decide (0 < s .wait) && s .commit == d - 1 + s .wait)]
  | .wgmmaCommit c => [("wgmma commit number", c + 1 == s .bfence && s .bcommit == c)]
  | .wgmmaWait0 c => [("wgmma wait number", c + 1 == s .bcommit && s .bwait == c)]
  | .settle =>
    [("settle before the last step", s .wait == count && s .barrier == count),
     ("settle before the last commit", s .commit == d - 1 + count),
     ("settle with an open wgmma batch", s .bfence == s .bwait)]

def obeys (d count : Nat) (proxy : Bool) (s : Counts) (prev : Option Instr) (x : Instr) : Bool :=
  (rules d count proxy s prev x).all (·.2)

def walk (d count : Nat) (proxy : Bool) :
    Counts → Option Instr → List Instr → Option (Counts × Option Instr)
  | s, prev, [] => some (s, prev)
  | s, prev, x :: xs =>
    if obeys d count proxy s prev x then walk d count proxy (s.add x) (some x) xs else none

/-- The checker for one thread's program-order sequence. -/
def accepts (d count : Nat) (proxy : Bool) (P : List Instr) : Bool :=
  decide (2 ≤ d) && match walk d count proxy (counts []) none P with
    | some (s, _) => s .settle == 1
    | none => false

/-! ### The canonical sequence -/

def primeOps (d count : Nat) : Nat → List Instr
  | 0 => []
  | j + 1 => primeOps d count j ++ [.fill j (j % d) (source d count j), .commit]

/-- Step `i` with `loads` synchronous reads and, under `proxy`, one wgmma batch of `mmas` reads. -/
def stepOps (d count : Nat) (proxy : Bool) (loads mmas i : Nat) : List Instr :=
  [.waitGroup (d - 2)] ++ (if proxy then [.proxyFence] else []) ++
  [.barrier, .fill (i + (d - 1)) ((i + (d - 1)) % d) (source d count (i + (d - 1))), .commit] ++
  List.replicate loads (.read i (i % d) .load) ++
  (if proxy then
    .wgmmaFence i :: List.replicate mmas (.read i (i % d) (.wgmma i)) ++ [.wgmmaCommit i, .wgmmaWait0 i]
   else [])

def stepsOps (d count : Nat) (proxy : Bool) (loads mmas : Nat) : Nat → List Instr
  | 0 => []
  | i + 1 => stepsOps d count proxy loads mmas i ++ stepOps d count proxy loads mmas i

def expected (d count : Nat) (proxy : Bool) (loads mmas : Nat) : List Instr :=
  primeOps d count (d - 1) ++ stepsOps d count proxy loads mmas count ++ [.settle]

theorem walk_append : walk d count proxy s p (A ++ B) =
    match walk d count proxy s p A with
    | some (s', p') => walk d count proxy s' p' B
    | none => none := by
  induction A generalizing s p with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, walk]; split <;> simp [ih]

private theorem walk_replicate (pass : ∀ p, obeys d count proxy s p x = true)
    (still : x.kind? = none) : ∃ p', walk d count proxy s p (List.replicate n x) = some (s, p') := by
  induction n generalizing p with
  | zero => exact ⟨p, rfl⟩
  | succ n ih =>
    have same : s.add x = s := by funext k; simp [Counts.add, still]
    simp only [List.replicate_succ, walk, pass, ite_true, same]
    exact ih

private theorem walk_then (h : walk d count proxy s p A = some (s', p'))
    (k : ∃ q, walk d count proxy s' p' B = some (s'', q)) :
    ∃ q, walk d count proxy s p (A ++ B) = some (s'', q) := by
  rw [walk_append, h]; exact k

private def stage (proxy : Bool) (d i : Nat) : Counts
  | .commit => d - 1 + i
  | .wait | .barrier => i
  | .proxy | .bfence | .bcommit | .bwait => if proxy then i else 0
  | .settle => 0

private def primed (j : Nat) : Counts
  | .commit => j
  | _ => 0

private theorem walk_prime (n : Nat) (hn : n ≤ d - 1) :
    ∃ p, walk d count proxy (counts []) none (primeOps d count n) = some (primed n, p) := by
  induction n with
  | zero => exact ⟨none, by simp [walk, primeOps]; funext k; cases k <;> rfl⟩
  | succ n ih =>
    obtain ⟨p, h⟩ := ih (by omega)
    apply walk_then h
    have src : source d count n = n := by unfold source; split <;> omega
    refine ⟨some .commit, ?_⟩
    simp [walk, obeys, rules, primed, Counts.add, Instr.kind?, src, isFill]
    refine ⟨⟨by omega, by omega⟩, ?_⟩
    funext k; cases k <;> simp [primed, Counts.add, Instr.kind?]

private theorem walk_seq (h : ∃ p', walk d count proxy s p A = some (s', p'))
    (k : ∀ p', ∃ q, walk d count proxy s' p' B = some (s'', q)) :
    ∃ q, walk d count proxy s p (A ++ B) = some (s'', q) := by
  obtain ⟨p', h⟩ := h
  rw [walk_append, h]; exact k p'

private def refilled (proxy : Bool) (d i : Nat) : Counts
  | .commit => d + i
  | .wait | .barrier => i + 1
  | .proxy => if proxy then i + 1 else 0
  | .bfence | .bcommit | .bwait => if proxy then i else 0
  | .settle => 0

private theorem walk_head (hd : 2 ≤ d) (hi : i < count) (p : Option Instr) :
    walk d count proxy (stage proxy d i) p
      ([.waitGroup (d - 2)] ++ (if proxy then [.proxyFence] else []) ++
        [.barrier, .fill (i + (d - 1)) ((i + (d - 1)) % d) (source d count (i + (d - 1))), .commit]) =
      some (refilled proxy d i, some .commit) := by
  cases proxy <;>
  · simp [walk, obeys, rules, stage, Counts.add, Instr.kind?, isFill, hi]
    refine And.intro (And.intro (by omega) (by omega)) ?_
    funext k; cases k <;> simp [refilled, stage, Counts.add, Instr.kind?] <;> omega

private theorem read_obeys (hd : 2 ≤ d) (p : Option Instr) :
    obeys d count proxy (refilled proxy d i) p (.read i (i % d) .load) = true := by
  simp [obeys, rules, refilled]; omega

private theorem walk_batch (hd : 2 ≤ d) (p : Option Instr) :
    ∃ q, walk d count true (refilled true d i) p
      (.wgmmaFence i :: List.replicate mmas (.read i (i % d) (.wgmma i)) ++
        [.wgmmaCommit i, .wgmmaWait0 i]) = some (stage true d (i + 1), q) := by
  let fenced : Counts := (refilled true d i).add (.wgmmaFence i)
  have fence : walk d count true (refilled true d i) p [.wgmmaFence i] = some (fenced, some (.wgmmaFence i)) := by
    simp [walk, obeys, rules, refilled, fenced]; omega
  rw [List.cons_append, ← List.singleton_append]
  apply walk_seq ⟨_, fence⟩
  intro p
  apply walk_seq (walk_replicate (fun p => by simp [obeys, rules, fenced, refilled, Counts.add, Instr.kind?]; omega) rfl)
  intro p
  refine ⟨some (.wgmmaWait0 i), ?_⟩
  simp [walk, obeys, rules, fenced, refilled, Counts.add, Instr.kind?]
  funext k; cases k <;> simp [stage, refilled, Counts.add, Instr.kind?] <;> omega

private theorem walk_step (hd : 2 ≤ d) (hi : i < count) (p : Option Instr) :
    ∃ q, walk d count proxy (stage proxy d i) p (stepOps d count proxy loads mmas i) =
      some (stage proxy d (i + 1), q) := by
  unfold stepOps
  rw [List.append_assoc]
  apply walk_seq ⟨_, walk_head hd hi p⟩
  intro p
  apply walk_seq (walk_replicate (fun p => read_obeys hd p) rfl)
  intro p
  cases proxy with
  | false =>
    refine ⟨p, ?_⟩
    simp only [Bool.false_eq_true, ite_false, walk, Option.some.injEq, Prod.mk.injEq, and_true]
    funext k; cases k <;> simp [refilled, stage] <;> omega
  | true => exact walk_batch hd p

theorem expected_accepted (hd : 2 ≤ d) :
    accepts d count proxy (expected d count proxy loads mmas) = true := by
  have steps (n : Nat) (hn : n ≤ count) :
      ∀ p, ∃ q, walk d count proxy (stage proxy d 0) p (stepsOps d count proxy loads mmas n) =
        some (stage proxy d n, q) := by
    induction n with
    | zero => intro p; exact ⟨p, rfl⟩
    | succ n ih =>
      intro p
      exact walk_seq (ih (by omega) p) (walk_step hd (by omega))
  obtain ⟨p, prime⟩ := walk_prime (d := d) (count := count) (proxy := proxy) (d - 1) (Nat.le_refl _)
  have start : primed (d - 1) = stage proxy d 0 := by
    funext k; cases k <;> simp [primed, stage]
  rw [start] at prime
  obtain ⟨q, all⟩ := walk_seq ⟨p, prime⟩ (steps count (Nat.le_refl _))
  unfold accepts expected
  rw [walk_append, all]
  simp [walk, obeys, rules, stage, hd, Counts.add, Instr.kind?]

/-! ### Occurrences in program order -/

variable {P pre rest : List Instr} {x : Instr}

theorem counts_snoc : counts (P ++ [x]) = (counts P).add x := by
  funext k; simp [counts, Counts.add, List.countP_append]

theorem walk_spec (h : walk d count proxy (counts pre) pre.getLast? rest = some r) :
    (∀ q x, rest[q]? = some x →
      obeys d count proxy (counts (pre ++ rest.take q)) (pre ++ rest.take q).getLast? x = true) ∧
    r.1 = counts (pre ++ rest) := by
  induction rest generalizing pre with
  | nil => simp [walk] at h; rw [← h]; simp
  | cons x xs ih =>
    simp only [walk] at h
    split at h
    · rename_i ok
      have next := ih (pre := pre ++ [x]) (by simpa [counts_snoc] using h)
      refine ⟨fun q y hy => ?_, by simpa using next.2⟩
      cases q with
      | zero => simp at hy; subst hy; simpa using ok
      | succ q => simpa using next.1 q y (by simpa using hy)
    · simp at h

/-- What a successful check guarantees about every position of `P`. -/
structure Accepted (d count : Nat) (proxy : Bool) (P : List Instr) : Prop where
  depth : 2 ≤ d
  rule : ∀ q x, P[q]? = some x →
    obeys d count proxy (counts (P.take q)) (P.take q).getLast? x = true
  settled : counts P .settle = 1

theorem accepted (h : accepts d count proxy P = true) : Accepted d count proxy P := by
  unfold accepts at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hd, h⟩ := h
  split at h
  · rename_i s p hw
    have spec := walk_spec (pre := []) (by simpa using hw)
    simp only [List.nil_append] at spec
    exact ⟨hd, spec.1, by rw [← spec.2]; simpa using h⟩
  · simp at h

theorem counts_take_succ (P : List Instr) (q : Nat) (k : Kind) :
    counts (P.take (q + 1)) k =
      counts (P.take q) k + if P[q]?.bind Instr.kind? = some k then 1 else 0 := by
  rw [List.take_add_one]
  cases h : P[q]? <;> simp [counts, List.countP_append]

theorem counts_take_mono (P : List Instr) (k : Kind) (h : q ≤ q') :
    counts (P.take q) k ≤ counts (P.take q') k := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => rw [counts_take_succ]; omega

theorem len_of (h : P[q]? = some x) : q < P.length := by
  apply Classical.byContradiction
  intro no
  simp [List.getElem?_eq_none (by omega : P.length ≤ q)] at h

/-- Position of the `n`-th instruction of kind `k`, counted from zero. -/
def nth (k : Kind) (P : List Instr) (n : Nat) : Option Nat :=
  (List.range P.length).find? fun q => P[q]?.bind Instr.kind? == some k && counts (P.take q) k == n

theorem nth_spec (h : nth k P n = some q) :
    P[q]?.bind Instr.kind? = some k ∧ counts (P.take q) k = n := by
  have := (List.find?_range_eq_some.mp h).1
  simpa using this

private theorem exists_at (h : n < counts (P.take p) k) :
    ∃ q, q < p ∧ P[q]?.bind Instr.kind? = some k ∧ counts (P.take q) k = n := by
  induction p with
  | zero => simp [counts] at h
  | succ p ih =>
    rw [counts_take_succ] at h
    by_cases lt : n < counts (P.take p) k
    · obtain ⟨q, hq, rest⟩ := ih lt; exact ⟨q, by omega, rest⟩
    · split at h
      · exact ⟨p, by omega, ‹_›, by omega⟩
      · omega

theorem nth_lt (h : n < counts (P.take p) k) : ∃ q, nth k P n = some q ∧ q < p := by
  obtain ⟨q, hq, kind, cnt⟩ := exists_at h
  cases e : nth k P n with
  | none =>
    obtain ⟨x, hx, _⟩ := Option.bind_eq_some_iff.mp kind
    have := List.find?_eq_none.mp e q (List.mem_range.mpr (len_of hx))
    simp [kind, cnt] at this
  | some q' =>
    refine ⟨q', rfl, ?_⟩
    have := (nth_spec e).2
    apply Classical.byContradiction
    intro le
    have := counts_take_mono P k (by omega : p ≤ q')
    omega

theorem le_nth (h : nth k P n = some q) (le : counts (P.take p) k ≤ n) : p ≤ q := by
  obtain ⟨kind, cnt⟩ := nth_spec h
  apply Classical.byContradiction
  intro lt
  have mono := counts_take_mono P k (by omega : q + 1 ≤ p)
  rw [counts_take_succ, kind] at mono
  simp at mono
  omega

theorem nth_instr (h : nth k P n = some q) : ∃ x, P[q]? = some x ∧ x.kind? = some k :=
  Option.bind_eq_some_iff.mp (nth_spec h).1

def Instr.ticket? : Instr → Option Nat
  | .fill j _ _ => some j
  | _ => none

/-- The first `cp.async` of ticket `j` starts overwriting its slot. -/
def firstFill (P : List Instr) (j : Nat) : Option Nat :=
  (List.range P.length).find? fun q => P[q]?.bind Instr.ticket? == some j

theorem firstFill_spec (h : firstFill P j = some q) : ∃ s src, P[q]? = some (.fill j s src) := by
  have := (List.find?_range_eq_some.mp h).1
  simp only [beq_iff_eq, Option.bind_eq_some_iff] at this
  obtain ⟨x, hx, t⟩ := this
  cases x <;> simp [Instr.ticket?] at t
  subst t
  exact ⟨_, _, hx⟩

theorem firstFill_le (h : P[q]? = some (.fill j s src)) : ∃ q', firstFill P j = some q' ∧ q' ≤ q := by
  cases e : firstFill P j with
  | none =>
    have := List.find?_eq_none.mp e q (List.mem_range.mpr (len_of h))
    simp [h, Instr.ticket?] at this
  | some q' =>
    refine ⟨q', rfl, ?_⟩
    have := (List.find?_range_eq_some.mp e).2.2 q
    apply Classical.byContradiction
    intro lt
    have := this (by omega)
    simp [h, Instr.ticket?] at this

/-! ### Rules at a position -/

section
variable (acc : Accepted d count proxy P)
include acc

theorem Accepted.fill (h : P[q]? = some (.fill j s src)) :
    j = counts (P.take q) .commit ∧ j < d - 1 + count ∧ s = j % d ∧ src = source d count j ∧
    (d - 1 ≤ j → counts (P.take q) .barrier = j - (d - 1) + 1) := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  obtain ⟨-, h1, h2, h3, h4, h5⟩ := r
  refine ⟨h1, h2, h3, h4, fun le => ?_⟩
  simp only [show ¬ j < d - 1 by omega, ite_false] at h5
  omega

theorem Accepted.commit (h : P[q]? = some Instr.commit) :
    0 < q ∧ ∃ j s src, P[q - 1]? = some (.fill j s src) := by
  have r := acc.rule q _ h
  simp [obeys, rules, List.getLast?_take] at r
  obtain ⟨-, r⟩ := r
  by_cases hq : q = 0
  · simp [hq, isFill] at r
  · have lt := len_of h
    obtain ⟨y, hy⟩ : ∃ y, P[q - 1]? = some y := ⟨_, List.getElem?_eq_getElem (by omega)⟩
    simp only [hq, ite_false, hy] at r
    refine ⟨by omega, ?_⟩
    cases y <;> simp [isFill] at r
    exact ⟨_, _, _, hy⟩

theorem Accepted.wait (h : P[q]? = some (Instr.waitGroup n)) :
    n = d - 2 ∧ counts (P.take q) .wait < count ∧
    counts (P.take q) .commit = d - 1 + counts (P.take q) .wait ∧
    counts (P.take q) .bfence = counts (P.take q) .bwait := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  obtain ⟨-, h1, h2, h3, -, h5⟩ := r
  exact ⟨h1, h2, h3, h5⟩

theorem Accepted.proxyFence (h : P[q]? = some Instr.proxyFence) :
    proxy = true ∧ counts (P.take q) .wait = counts (P.take q) .proxy + 1 := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  exact ⟨r.2.1, r.2.2.1⟩

theorem Accepted.barrier (h : P[q]? = some Instr.barrier) :
    counts (P.take q) .barrier + 1 = counts (P.take q) .wait ∧
    (proxy = true → counts (P.take q) .proxy = counts (P.take q) .wait) := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  obtain ⟨-, h1, h2⟩ := r
  refine ⟨h1, fun hp => ?_⟩
  simpa [hp] using h2

theorem Accepted.read (h : P[q]? = some (Instr.read b s rd)) :
    b + 1 = counts (P.take q) .wait ∧ counts (P.take q) .commit = d - 1 + counts (P.take q) .wait ∧
    s = b % d ∧ counts (P.take q) .settle = 0 ∧
    ∀ c, rd = .wgmma c → proxy = true ∧ c + 1 = counts (P.take q) .bfence ∧
      counts (P.take q) .bcommit = c := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  obtain ⟨h0, h1, h2, h3, h4⟩ := r
  refine ⟨h1, h2, h3, h0, fun c hc => ?_⟩
  subst hc
  simp at h4
  exact ⟨h4.1.1, h4.1.2, h4.2⟩

theorem Accepted.bwait (h : P[q]? = some (Instr.wgmmaWait0 c)) :
    c + 1 = counts (P.take q) .bcommit ∧ counts (P.take q) .bwait = c := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  exact r.2

theorem Accepted.settle (h : P[q]? = some Instr.settle) :
    counts (P.take q) .wait = count ∧ counts (P.take q) .commit = d - 1 + count ∧
    counts (P.take q) .bfence = counts (P.take q) .bwait := by
  have r := acc.rule q _ h
  simp [obeys, rules] at r
  exact ⟨r.2.1.1, r.2.2⟩

end

theorem nth_commit (h : nth .commit P n = some q) : P[q]? = some .commit := by
  obtain ⟨x, hx, k⟩ := nth_instr h; cases x <;> simp [Instr.kind?] at k; exact hx

theorem nth_wait (h : nth .wait P n = some q) : ∃ m, P[q]? = some (.waitGroup m) := by
  obtain ⟨x, hx, k⟩ := nth_instr h; cases x <;> simp [Instr.kind?] at k; exact ⟨_, hx⟩

theorem nth_proxy (h : nth .proxy P n = some q) : P[q]? = some .proxyFence := by
  obtain ⟨x, hx, k⟩ := nth_instr h; cases x <;> simp [Instr.kind?] at k; exact hx

theorem nth_barrier (h : nth .barrier P n = some q) : P[q]? = some .barrier := by
  obtain ⟨x, hx, k⟩ := nth_instr h; cases x <;> simp [Instr.kind?] at k; exact hx

theorem nth_settle (h : nth .settle P n = some q) : P[q]? = some .settle := by
  obtain ⟨x, hx, k⟩ := nth_instr h; cases x <;> simp [Instr.kind?] at k; exact hx

theorem nth_bwait (h : nth .bwait P n = some q) : ∃ c, P[q]? = some (.wgmmaWait0 c) := by
  obtain ⟨x, hx, k⟩ := nth_instr h; cases x <;> simp [Instr.kind?] at k; exact ⟨_, hx⟩

theorem Accepted.fillBeforeCommit (acc : Accepted d count proxy P) (h : nth .commit P j = some q) :
    ∃ p, firstFill P j = some p ∧ p < q := by
  have cnt := (nth_spec h).2
  obtain ⟨pos, j', s, src, hf⟩ := acc.commit (nth_commit h)
  have tk := (acc.fill hf).1
  have step := counts_take_succ P (q - 1) .commit
  rw [show q - 1 + 1 = q by omega, hf] at step
  simp [Instr.kind?] at step
  have same : j' = j := by omega
  subst same
  obtain ⟨p, hp, le⟩ := firstFill_le hf
  exact ⟨p, hp, by omega⟩

theorem Accepted.fillExists (acc : Accepted d count proxy P) (hj : j < d - 1 + count) :
    ∃ q s src, firstFill P j = some q ∧ P[q]? = some (.fill j s src) := by
  obtain ⟨qs, hs, -⟩ := nth_lt (k := .settle) (n := 0) (p := P.length) (P := P)
    (by rw [List.take_length, acc.settled]; omega)
  have commits := (acc.settle (nth_settle hs)).2.1
  obtain ⟨qc, hc, -⟩ := nth_lt (k := .commit) (n := j) (P := P) (p := qs) (by omega)
  obtain ⟨q, hq, -⟩ := acc.fillBeforeCommit hc
  obtain ⟨s, src, h⟩ := firstFill_spec hq
  exact ⟨q, s, src, hq, h⟩

/-! ### Timelines consistent with program order -/

/-- Thread `l` has executed the first `done l` instructions of its program,
    instruction `q` at `time l q`. Device and barrier events carry their own times. -/
structure Exec (threads : Nat) where
  done : Fin threads → Nat
  time : Fin threads → Nat → Nat
  release : Nat → Event
  complete : Fin threads → Nat → Event
  batchDone : Fin threads → Nat → Event

def Exec.at (e : Exec threads) (l : Fin threads) : Option Nat → Event
  | some q => event (e.done l) q (e.time l q)
  | none => none

/-- Program order, CTA barrier semantics and the hardware rules. A thread's
    `k`-th `bar.sync` must arrive before release `k`, and its next instruction
    follows the release. Wait rules refer to the thread's own commits. -/
structure Consistent (prog : Fin threads → List Instr) (e : Exec threads) : Prop where
  order : ∀ l p q, p < q → q < e.done l → e.time l p < e.time l q
  arrive : ∀ l k, e.at l (nth .barrier (prog l) k) ≺ e.release k
  pass : ∀ l k q, nth .barrier (prog l) k = some q → e.release k ≺ e.at l (some (q + 1))
  commitDone : ∀ l j, e.at l (nth .commit (prog l) j) ≺ e.complete l j
  waitGroup : ∀ l q n, (prog l)[q]? = some (.waitGroup n) → ∀ j t, e.at l (some q) = some t →
    By (e.at l (nth .commit (prog l) (j + n))) t → By (e.complete l j) t
  settle : ∀ l q j, (prog l)[q]? = some .settle →
    e.at l (nth .commit (prog l) j) ≺ e.at l (some q) → e.complete l j ≺ e.at l (some q)
  batchDone : ∀ l b, e.at l (nth .bcommit (prog l) b) ≺ e.batchDone l b
  batchWait : ∀ l q c b, (prog l)[q]? = some (.wgmmaWait0 c) →
    e.at l (nth .bcommit (prog l) b) ≺ e.at l (some q) → e.batchDone l b ≺ e.at l (some q)

def fillOf (P : List Instr) (j : Nat) : Option Instr := (firstFill P j).bind (P[·]?)

def stream (prog : Fin threads → List Instr) (e : Exec threads) (l : Fin threads) : Stream where
  slot j := match fillOf (prog l) j with | some (.fill _ s _) => s | _ => 0
  sourceBlock j := match fillOf (prog l) j with | some (.fill _ _ src) => src | _ => 0
  issue j := e.at l (firstFill (prog l) j)
  commit j := e.at l (nth .commit (prog l) j)
  complete := e.complete l
  waitGroup i := e.at l (nth .wait (prog l) i)
  proxyFence i := e.at l (nth .proxy (prog l) i)
  settle := e.at l (nth .settle (prog l) 0)

def batchEvent (prog : Fin threads → List Instr) (e : Exec threads) (k : Kind) (w b : Nat) : Event :=
  if h : w < threads then e.at ⟨w, h⟩ (nth k (prog ⟨w, h⟩) b) else none

/-- Each thread stands for its warpgroup; its own wgmma events are the batch's. -/
def cta (prog : Fin threads → List Instr) (e : Exec threads) : CTA threads where
  writer := stream prog e
  fence := e.release
  warpgroup l := l.val
  batchFence := batchEvent prog e .bfence
  batchCommit := batchEvent prog e .bcommit
  batchDone w b := if h : w < threads then e.batchDone ⟨w, h⟩ b else none
  batchWait := batchEvent prog e .bwait

/-- The read at position `q` of thread `l`, if any. -/
def readAt (prog : Fin threads → List Instr) (e : Exec threads) (l : Fin threads) (q : Nat) :
    Option (Read threads) :=
  match (prog l)[q]? with
  | some (.read b s rd) => some ⟨l, b, s, e.at l (some q), rd⟩
  | _ => none

section
variable {prog : Fin threads → List Instr} {e : Exec threads}

@[simp] theorem batchEvent_lane (l : Fin threads) :
    batchEvent prog e k l.val b = e.at l (nth k (prog l) b) := by
  simp [batchEvent, l.isLt]

theorem Consistent.lt (con : Consistent prog e) (h : p < q) : e.at l (some p) ≺ e.at l (some q) := by
  simp only [Exec.at, event_edge]
  intro hq
  exact ⟨by omega, con.order l p q h hq⟩

theorem Consistent.later (con : Consistent prog e) (ab : a ≺ e.at l (some p)) (h : p ≤ q) :
    a ≺ e.at l (some q) := by
  rcases Nat.eq_or_lt_of_le h with rfl | lt
  · exact ab
  · exact ab.trans (con.lt lt)

theorem Consistent.ordered (con : Consistent prog e)
    (h : ∀ q, o' = some q → ∃ p, o = some p ∧ p < q) : e.at l o ≺ e.at l o' := by
  cases o' with
  | none => intro t ht; simp [Exec.at] at ht
  | some q => obtain ⟨p, rfl, lt⟩ := h q rfl; exact con.lt lt

theorem edge_none_left {b : Event} (h : (none : Event) ≺ b) : b = none := by
  cases b with
  | none => rfl
  | some t => obtain ⟨u, hu, _⟩ := h t rfl; cases hu

theorem edge_none_right {a : Event} : a ≺ (none : Event) := by intro t ht; cases ht

theorem fold_of_accepted (acc : ∀ l, Accepted d count proxy (prog l)) (con : Consistent prog e) :
    Fold (cta prog e) d count (d - 2) proxy where
  fillSlot l j hj := by
    obtain ⟨q, s, src, hf, hq⟩ := (acc l).fillExists hj
    simp [cta, stream, fillOf, hf, hq, ((acc l).fill hq).2.2.1]
  fillSource l j hj := by
    obtain ⟨q, s, src, hf, hq⟩ := (acc l).fillExists hj
    simp [cta, stream, fillOf, hf, hq, ((acc l).fill hq).2.2.2.1]
  issueCommit l j := con.ordered fun q h => (acc l).fillBeforeCommit h
  commitDone l j := con.commitDone l j
  commitOrder l j _ := con.ordered fun q h => by
    obtain ⟨s, src, hq⟩ := firstFill_spec h
    obtain ⟨p, hp, lt⟩ := nth_lt (k := .commit) (n := j) (p := q) (P := prog l)
      (by have := ((acc l).fill hq).1; omega)
    exact ⟨p, hp, lt⟩
  priorCommit l i _ := con.ordered fun q h => by
    obtain ⟨m, hq⟩ := nth_wait h
    have w := (acc l).wait hq
    have hd := (acc l).depth
    have := (nth_spec h).2
    exact nth_lt (by omega)
  waitRule l i _ j t ht := by
    cases hw : nth .wait (prog l) i with
    | none => simp [cta, stream, hw, Exec.at] at ht
    | some q =>
      obtain ⟨m, hq⟩ := nth_wait hw
      have := ((acc l).wait hq).1
      subst this
      simp only [cta, stream, hw] at ht ⊢
      exact con.waitGroup l q _ hq j t ht
  publish l i _ _ := con.ordered fun q h => by
    have w := ((acc l).proxyFence (nth_proxy h)).2
    have := (nth_spec h).2
    exact nth_lt (by omega)
  arrive l i _ := by
    have h := con.arrive l i
    cases hb : nth .barrier (prog l) i with
    | none =>
      rw [hb] at h
      simp only [Exec.at] at h
      simp only [cta, edge_none_left h]
      exact edge_none_right
    | some qb =>
      rw [hb] at h
      have bar := (acc l).barrier (nth_barrier hb)
      have cnt := (nth_spec hb).2
      refine Edge.trans ?_ h
      cases proxy with
      | false =>
        obtain ⟨p, hp, lt⟩ := nth_lt (k := .wait) (n := i) (P := prog l) (p := qb) (by omega)
        simp only [cta, stream, hp, ite_false, Bool.false_eq_true]
        exact con.lt lt
      | true =>
        obtain ⟨p, hp, lt⟩ := nth_lt (k := .proxy) (n := i) (P := prog l) (p := qb)
          (by have := bar.2 rfl; omega)
        simp only [cta, stream, hp, ite_true]
        exact con.lt lt
  refill l i _ := by
    cases hf : firstFill (prog l) (i + (d - 1)) with
    | none => simp only [cta, stream, hf, Exec.at]; exact edge_none_right
    | some q =>
      obtain ⟨s, src, hq⟩ := firstFill_spec hf
      have bar := ((acc l).fill hq).2.2.2.2 (by omega)
      obtain ⟨qb, hb, lt⟩ := nth_lt (k := .barrier) (n := i) (P := prog l) (p := q) (by omega)
      simp only [cta, stream, hf]
      exact con.later (con.pass l i qb hb) lt
  drain l j _ := by
    cases hs : nth .settle (prog l) 0 with
    | none => simp only [cta, stream, hs, Exec.at]; exact edge_none_right
    | some q =>
      have st := (acc l).settle (nth_settle hs)
      obtain ⟨qc, hc, lt⟩ := nth_lt (k := .commit) (n := j) (P := prog l) (p := q) (by omega)
      simp only [cta, stream, hs]
      apply con.settle l q j (nth_settle hs)
      rw [hc]
      exact con.lt lt

theorem nth_ne (h : nth k P n = some q') (hq : P[q]? = some (.read b s rd)) : q ≠ q' := by
  rintro rfl
  have := (nth_spec h).1
  simp [hq, Instr.kind?] at this

private theorem Accepted.nextStep (acc : Accepted d count proxy P)
    (hq : P[q]? = some (.read b s rd))
    (hn : (if b + 1 < count then nth .wait P (b + 1) else nth .settle P 0) = some q') :
    q < q' ∧ counts (P.take q') .bfence = counts (P.take q') .bwait := by
  have r := acc.read hq
  split at hn
  · obtain ⟨m, hw⟩ := nth_wait hn
    have le := le_nth hn (p := q) (by omega)
    exact ⟨by have := nth_ne hn hq; omega, ((acc.wait hw).2.2.2)⟩
  · have le := le_nth hn (p := q) (by omega)
    exact ⟨by have := nth_ne hn hq; omega, (acc.settle (nth_settle hn)).2.2⟩

private theorem afterStep_eq (r : Read threads) :
    afterStep (cta prog e) count r =
      e.at r.lane (if r.block + 1 < count then nth .wait (prog r.lane) (r.block + 1)
        else nth .settle (prog r.lane) 0) := by
  unfold afterStep; split <;> rfl

theorem readCode_of_accepted (acc : ∀ l, Accepted d count proxy (prog l)) (con : Consistent prog e)
    (hr : readAt prog e l q = some r) : ReadCode (cta prog e) d count proxy r := by
  unfold readAt at hr
  split at hr
  case h_2 => cases hr
  rename_i b s rd hq
  cases hr
  have rr := (acc l).read hq
  have hd := (acc l).depth
  refine ⟨?_, rr.2.2.1, ?_, ?_, ?_⟩
  · show b < count
    obtain ⟨qw, hw, lt⟩ := nth_lt (k := .wait) (n := b) (P := prog l) (p := q) (by omega)
    obtain ⟨m, hm⟩ := nth_wait hw
    have := ((acc l).wait hm).2.1
    have := (nth_spec hw).2
    omega
  · show e.at l (nth .commit (prog l) (b + (d - 1))) ≺ e.at l (some q)
    exact con.ordered fun q' h => by cases h; exact nth_lt (by omega)
  · intro c hc
    obtain ⟨hp, hf, -⟩ := rr.2.2.2.2 c hc
    refine ⟨hp, ?_⟩
    simp only [cta, batchEvent_lane]
    exact con.ordered fun q' h => by cases h; exact nth_lt (by omega)
  · have after := afterStep_eq (prog := prog) (e := e) (count := count) ⟨l, b, s, e.at l (some q), rd⟩
    cases rd with
    | load =>
      simp only [after]
      exact con.ordered fun q' h => ⟨q, rfl, ((acc l).nextStep hq h).1⟩
    | wgmma c =>
      obtain ⟨-, hf, hc⟩ := rr.2.2.2.2 c rfl
      simp only [after]
      simp only [cta, batchEvent_lane, l.isLt, dite_true]
      refine ⟨?_, con.batchDone l c, ?_, ?_⟩
      · exact con.ordered fun q' h => ⟨q, rfl, by
          have := le_nth h (p := q) (by omega); have := nth_ne h hq; omega⟩
      · cases hw : nth .bwait (prog l) c with
        | none => simp only [Exec.at]; exact edge_none_right
        | some qw =>
          obtain ⟨c', hc'⟩ := nth_bwait hw
          have bw := (acc l).bwait hc'
          have := (nth_spec hw).2
          apply con.batchWait l qw c' c hc'
          exact con.ordered fun q' h => by
            cases h; exact nth_lt (by omega)
      · exact con.ordered fun q' h => by
          obtain ⟨lt, eq⟩ := (acc l).nextStep hq h
          have := counts_take_mono (prog l) .bfence (Nat.le_of_lt lt)
          exact nth_lt (k := .bwait) (n := c) (p := q') (P := prog l) (by omega)

/-- T1 for every read of any accepted program, in any consistent timeline. -/
theorem accepted_read_current_generation (acc : ∀ l, accepts d count proxy (prog l) = true)
    (con : Consistent prog e) (hr : readAt prog e l' q = some r) (l : Fin threads)
    (issued : By r.issue cut) (active : r.issue = some cut ∨ ¬ By (retired (cta prog e) r) cut) :
    (writes (cta prog e) d count l).Shows r.slot r.block r.block cut :=
  have acc' := fun l => accepted (acc l)
  read_current_generation (fold_of_accepted acc' con) (acc' l).depth
    (readCode_of_accepted acc' con hr) l issued active

/-- T2 for every read of any accepted program and every refill of any thread. -/
theorem accepted_overwrite_after_last_read (acc : ∀ l, accepts d count proxy (prog l) = true)
    (con : Consistent prog e) (hr : readAt prog e l' q = some r) (l : Fin threads)
    (bound : k < d - 1 + count) (later : r.block < k)
    (same : ((cta prog e).writer l).slot k = r.slot) :
    retired (cta prog e) r ≺ ((cta prog e).writer l).issue k :=
  have acc' := fun l => accepted (acc l)
  overwrite_after_last_read (fold_of_accepted acc' con) (acc' l).depth
    (readCode_of_accepted acc' con hr) l bound later same

end

/-! ### Diagnostics -/

/-- The first violated rule and its instruction index; index `P.length` refers to the end. -/
def diagnoseFrom (d count : Nat) (proxy : Bool) :
    Counts → Option Instr → Nat → List Instr → Option (Nat × String)
  | s, _, i, [] => if s .settle == 1 then none else some (i, "expected exactly one settle")
  | s, prev, i, x :: xs => match (rules d count proxy s prev x).find? (!·.2) with
    | some (msg, _) => some (i, msg)
    | none => diagnoseFrom d count proxy (s.add x) (some x) (i + 1) xs

def diagnose (d count : Nat) (proxy : Bool) (P : List Instr) : Option (Nat × String) :=
  if 2 ≤ d then diagnoseFrom d count proxy (counts []) none 0 P else some (0, "depth below 2")

private theorem diagnoseFrom_none (h : diagnoseFrom d count proxy s prev i P = none) :
    ∃ r, walk d count proxy s prev P = some r ∧ r.1 .settle = 1 := by
  induction P generalizing s prev i with
  | nil => simp [diagnoseFrom] at h; exact ⟨_, rfl, h⟩
  | cons x xs ih =>
    simp only [diagnoseFrom] at h
    split at h
    · cases h
    · rename_i none_
      have ok : obeys d count proxy s prev x = true := by
        simp only [obeys, List.all_eq_true]
        intro r hr
        have := List.find?_eq_none.mp none_ r hr
        simpa using this
      simp only [walk, ok, ite_true]
      exact ih h

/-- The executable's verdict `accepted` is the verified checker's. -/
theorem diagnose_none (h : diagnose d count proxy P = none) : accepts d count proxy P = true := by
  unfold diagnose at h
  split at h
  · obtain ⟨⟨s, p⟩, hw, hs⟩ := diagnoseFrom_none h
    simp only at hs
    simp [accepts, *]
  · cases h

/-! ### Rejections -/

variable {q b s j src n : Nat}

theorem wrong_slot_rejected (h : P[q]? = some (Instr.read b s rd)) (wrong : s ≠ b % d) :
    accepts d count proxy P = false :=
  Bool.eq_false_iff.mpr fun acc => wrong ((accepted acc).read h).2.2.1

theorem wrong_immediate_rejected (h : P[q]? = some (Instr.waitGroup n)) (wrong : n ≠ d - 2) :
    accepts d count proxy P = false :=
  Bool.eq_false_iff.mpr fun acc => wrong ((accepted acc).wait h).1

theorem refill_without_barrier_rejected (h : P[q]? = some (Instr.fill j s src)) (refill : d - 1 ≤ j)
    (few : counts (P.take q) .barrier ≤ j - (d - 1)) : accepts d count proxy P = false :=
  Bool.eq_false_iff.mpr fun acc => by
    have := ((accepted acc).fill h).2.2.2.2 refill
    omega

example : accepts 2 3 false (expected 2 3 false 2 0) = true := by decide
example : accepts 2 3 true (expected 2 3 true 1 2) = true := by decide

/-- Block 1 read from slot 0. -/
example : accepts 2 2 false
    [.fill 0 0 0, .commit, .waitGroup 0, .barrier, .fill 1 1 1, .commit, .read 0 0 .load,
     .waitGroup 0, .barrier, .fill 2 0 1, .commit, .read 1 0 .load, .settle] = false := by decide

/-- The second refill lacks its barrier. -/
example : accepts 2 2 false
    [.fill 0 0 0, .commit, .waitGroup 0, .barrier, .fill 1 1 1, .commit, .read 0 0 .load,
     .waitGroup 0, .fill 2 0 1, .commit, .read 1 1 .load, .settle] = false := by decide

/-- `cp.async.wait_group 1` lets the current block's group stay in flight. -/
example : accepts 2 2 false
    [.fill 0 0 0, .commit, .waitGroup 1, .barrier, .fill 1 1 1, .commit, .read 0 0 .load,
     .waitGroup 1, .barrier, .fill 2 0 1, .commit, .read 1 1 .load, .settle] = false := by decide

example : accepts 2 2 false
    [.fill 0 0 0, .commit, .waitGroup 0, .barrier, .fill 1 1 1, .commit, .read 0 0 .load,
     .waitGroup 0, .barrier, .fill 2 0 1, .commit, .read 1 1 .load, .settle] = true := by decide

end Ring.Async
