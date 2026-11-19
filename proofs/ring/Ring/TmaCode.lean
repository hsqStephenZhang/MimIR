import Ring.Phase

namespace Ring.Tma
open Timeline

/-- An mbarrier operand: the full or the empty barrier of a slot. -/
inductive Bar where
  | full (slot : Nat)
  | empty (slot : Nat)
  deriving DecidableEq, Repr

/-- One role's TMA ring instructions in program order, at PTX level.
    A fill is one `expect_tx` followed by one or more `tma_load`s on the same barrier. -/
inductive Instr where
  | init (bar : Bar) (arrivals : Nat)
  | expectTx (bar : Bar) (bytes : Nat)
  | tmaLoad (slot : Nat) (bar : Bar) (src bytes : Nat)
  | mbarWait (bar : Bar) (parity : Nat)
  | mbarArrive (bar : Bar)
  | barrier
  | wgmmaFence
  | read (block slot : Nat)
  | wgmmaCommit
  | wgmmaWait (pending : Nat)
  | settle
  deriving DecidableEq, Repr

/-- Instructions whose occurrences are numbered. -/
inductive Kind where
  | init | expect | fwait | ewait | arrive | barrier | bfence | bcommit | bwait | settle
  deriving DecidableEq, Repr

def Instr.kind? : Instr → Option Kind
  | .init .. => some .init
  | .expectTx .. => some .expect
  | .mbarWait (.full _) _ => some .fwait
  | .mbarWait (.empty _) _ => some .ewait
  | .mbarArrive _ => some .arrive
  | .barrier => some .barrier
  | .wgmmaFence => some .bfence
  | .wgmmaCommit => some .bcommit
  | .wgmmaWait _ => some .bwait
  | .settle => some .settle
  | .tmaLoad .. | .read .. => none

abbrev Counts := Kind → Nat

def counts (P : List Instr) : Counts := fun k => P.countP (·.kind? == some k)

@[simp] theorem counts_nil (k : Kind) : counts [] k = 0 := rfl

def Counts.add (s : Counts) (x : Instr) : Counts := fun k => s k + if x.kind? = some k then 1 else 0

def isLoad : Instr → Bool
  | .tmaLoad .. => true
  | _ => false

def inFill : Option Instr → Bool
  | some (.expectTx ..) | some (.tmaLoad ..) => true
  | _ => false

def afterExpect : Option Instr → Instr → Bool
  | some (.expectTx ..), x => isLoad x
  | _, _ => true

/-- Local rules of one role, evaluated against the counts of the preceding instructions. -/
abbrev Rules := Counts → Option Instr → Instr → List (String × Bool)

def checks (R : Rules) : Rules := fun s prev x =>
  ("instruction after settle", s .settle == 0) ::
  ("expect_tx not followed by tma_load", afterExpect prev x) :: R s prev x

def obeys (R : Rules) (s : Counts) (prev : Option Instr) (x : Instr) : Bool :=
  (checks R s prev x).all (·.2)

def walk (R : Rules) : Counts → Option Instr → List Instr → Option (Counts × Option Instr)
  | s, prev, [] => some (s, prev)
  | s, prev, x :: xs => if obeys R s prev x then walk R (s.add x) (some x) xs else none

def run (R : Rules) (P : List Instr) : Bool :=
  match walk R (counts []) none P with
  | some (s, _) => s .settle == 1
  | none => false

theorem walk_append : walk R s p (A ++ B) =
    match walk R s p A with
    | some (s', p') => walk R s' p' B
    | none => none := by
  induction A generalizing s p with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, walk]; split <;> simp [ih]

theorem walk_seq (h : ∃ p', walk R s p A = some (s', p'))
    (k : ∀ p', ∃ q, walk R s' p' B = some (s'', q)) :
    ∃ q, walk R s p (A ++ B) = some (s'', q) := by
  obtain ⟨p', h⟩ := h
  rw [walk_append, h]; exact k p'

theorem walk_replicate (pass : ∀ p, obeys R s p x = true) (still : x.kind? = none) :
    ∃ p', walk R s p (List.replicate n x) = some (s, p') := by
  induction n generalizing p with
  | zero => exact ⟨p, rfl⟩
  | succ n ih =>
    have same : s.add x = s := by funext k; simp [Counts.add, still]
    simp only [List.replicate_succ, walk, pass, ite_true, same]
    exact ih

theorem walk_while {Q : Option Instr → Prop} (pass : ∀ p, Q p → obeys R s p x = true)
    (keep : Q (some x)) (still : x.kind? = none) (hp : Q p) :
    ∃ p', walk R s p (List.replicate n x) = some (s, p') ∧ Q p' := by
  induction n generalizing p with
  | zero => exact ⟨p, rfl, hp⟩
  | succ n ih =>
    have same : s.add x = s := by funext k; simp [Counts.add, still]
    simp only [List.replicate_succ, walk, pass p hp, ite_true, same]
    exact ih keep

/-- The previous instruction leaves no fill without its first `tma_load`. -/
def closed : Option Instr → Bool
  | some (.expectTx ..) => false
  | _ => true

theorem afterExpect_closed (h : closed p = true) : afterExpect p x = true := by
  cases p with
  | none => rfl
  | some y => cases y <;> simp_all [closed, afterExpect]

/-- `L` takes the counts from `s` to `s'` after any closed instruction and ends closed. -/
def Steps (R : Rules) (s s' : Counts) (L : List Instr) : Prop :=
  ∀ p, closed p = true → ∃ q, walk R s p L = some (s', q) ∧ closed q = true

theorem Steps.append (a : Steps R s s' A) (b : Steps R s' s'' B) : Steps R s s'' (A ++ B) := by
  intro p hp
  obtain ⟨q, h, hq⟩ := a p hp
  rw [walk_append, h]
  exact b q hq

theorem Steps.nil : Steps R s s [] := fun p hp => ⟨p, rfl, hp⟩

theorem Steps.eq (a : Steps R s s' L) (h : s' = s'') : Steps R s s'' L := h ▸ a

theorem Steps.single (h : ∀ p, closed p = true → (R s p x).all (·.2) = true)
    (fresh : s .settle = 0) (hx : closed (some x) = true) : Steps R s (s.add x) [x] := by
  intro p hp
  refine ⟨some x, ?_, hx⟩
  have ok : obeys R s p x = true := by
    simp [obeys, checks, fresh, afterExpect_closed hp, h p hp]
  simp [walk, ok]

theorem Steps.cons (h : ∀ p, closed p = true → (R s p x).all (·.2) = true)
    (fresh : s .settle = 0) (hx : closed (some x) = true) (rest : Steps R (s.add x) s' L) :
    Steps R s s' (x :: L) := by
  rw [← List.singleton_append]
  exact (Steps.single h fresh hx).append rest

theorem run_of_steps (h : Steps R (counts []) s' P) (hs : s' .settle = 1) : run R P = true := by
  obtain ⟨q, hw, -⟩ := h none rfl
  simp [run, hw, hs]

theorem Steps.replicate (h : ∀ p, closed p = true → (R s p x).all (·.2) = true)
    (fresh : s .settle = 0) (hx : closed (some x) = true) (still : x.kind? = none) :
    Steps R s s (List.replicate n x) := by
  induction n with
  | zero => exact Steps.nil
  | succ n ih =>
    have same : s.add x = s := by funext k; simp [Counts.add, still]
    rw [List.replicate_succ, ← List.singleton_append]
    exact (same ▸ Steps.single h fresh hx).append ih

/-- A fill: its `expect_tx` checked after any closed instruction, its loads after the fill's previous op. -/
theorem Steps.fill (he : ∀ p, closed p = true → (R s p (.expectTx b n)).all (·.2) = true)
    (hl : ∀ p, inFill p = true → (R (s.add (.expectTx b n)) p (.tmaLoad sl b' src m)).all (·.2) = true)
    (fresh : s .settle = 0) (pos : 0 < k) :
    Steps R s (s.add (.expectTx b n)) (.expectTx b n :: List.replicate k (.tmaLoad sl b' src m)) := by
  intro p hp
  have ok : obeys R s p (.expectTx b n) = true := by
    simp [obeys, checks, fresh, afterExpect_closed hp, he p hp]
  have fresh' : (s.add (.expectTx b n)) .settle = 0 := by simp [Counts.add, Instr.kind?, fresh]
  have load (p) (hp : inFill p = true) :
      obeys R (s.add (.expectTx b n)) p (.tmaLoad sl b' src m) = true := by
    have : afterExpect p (.tmaLoad sl b' src m) = true := by
      cases p with
      | none => rfl
      | some y => cases y <;> rfl
    simp [obeys, checks, fresh', this, hl p hp]
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  simp only [walk, ok, ite_true, List.replicate_succ, load (some (.expectTx b n)) rfl]
  have same : (s.add (.expectTx b n)).add (.tmaLoad sl b' src m) = s.add (.expectTx b n) := by
    funext k; simp [Counts.add, Instr.kind?]
  obtain ⟨q, hq, hq'⟩ := walk_while (Q := fun p => p = some (.tmaLoad sl b' src m)) (n := k)
    (fun p hp => load p (by rw [hp]; rfl)) rfl rfl rfl
  rw [same, hq]
  exact ⟨q, rfl, by rw [hq']; rfl⟩

variable {P pre rest : List Instr} {x : Instr}

theorem counts_snoc : counts (P ++ [x]) = (counts P).add x := by
  funext k; simp [counts, Counts.add, List.countP_append]

theorem walk_spec (h : walk R (counts pre) pre.getLast? rest = some r) :
    (∀ q x, rest[q]? = some x →
      obeys R (counts (pre ++ rest.take q)) (pre ++ rest.take q).getLast? x = true) ∧
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

/-- What a successful run guarantees about every position of `P`. -/
structure Obeys (R : Rules) (P : List Instr) : Prop where
  rule : ∀ q x, P[q]? = some x → (R (counts (P.take q)) (P.take q).getLast? x).all (·.2) = true
  fresh : ∀ q x, P[q]? = some x → counts (P.take q) .settle = 0
  loads : ∀ q x, P[q]? = some x → afterExpect (P.take q).getLast? x = true
  settled : counts P .settle = 1

theorem obeys_of_run (h : run R P = true) : Obeys R P := by
  unfold run at h
  split at h
  · rename_i s p hw
    have spec := walk_spec (pre := []) (by simpa using hw)
    simp only [List.nil_append] at spec
    have each (q x) (hx : P[q]? = some x) := spec.1 q x hx
    refine ⟨fun q x hx => ?_, fun q x hx => ?_, fun q x hx => ?_, by rw [← spec.2]; simpa using h⟩
    · have := each q x hx; simp only [obeys, checks, List.all_cons] at this
      simp_all
    · have := each q x hx; simp only [obeys, checks, List.all_cons] at this
      simp_all
    · have := each q x hx; simp only [obeys, checks, List.all_cons] at this
      simp_all
  · simp at h

/-! ### Occurrences in program order -/

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

theorem counts_at (h : P[q]? = some x) (k : Kind) :
    counts (P.take (q + 1)) k = counts (P.take q) k + if x.kind? = some k then 1 else 0 := by
  rw [counts_take_succ, h]; rfl

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

theorem nth_earlier (h : nth k P n = some q') (lt : m < n) : ∃ q, nth k P m = some q ∧ q < q' :=
  nth_lt (by rw [(nth_spec h).2]; exact lt)

theorem nth_instr (h : nth k P n = some q) : ∃ x, P[q]? = some x ∧ x.kind? = some k :=
  Option.bind_eq_some_iff.mp (nth_spec h).1

theorem nth_at (h : P[q]? = some x) (kind : x.kind? = some k) : nth k P (counts (P.take q) k) = some q := by
  obtain ⟨q', hq', lt⟩ := nth_lt (k := k) (n := counts (P.take q) k) (p := q + 1) (P := P)
    (by rw [counts_at h, kind]; simp)
  have := le_nth hq' (p := q) (Nat.le_refl _)
  rw [show q' = q by omega] at hq'
  exact hq'

/-- An instruction of another kind sits strictly before or after the `n`-th of kind `k`. -/
theorem lt_nth (h : nth k P n = some q') (hq : P[q]? = some x) (other : x.kind? ≠ some k)
    (le : counts (P.take q) k ≤ n) : q < q' := by
  have := le_nth h le
  rcases Nat.eq_or_lt_of_le this with rfl | lt
  · obtain ⟨y, hy, ky⟩ := nth_instr h
    rw [hq] at hy; cases hy; exact (other ky).elim
  · exact lt

/-! ### Rules at a position -/

theorem kind_settle (h : x.kind? = some .settle) : x = .settle := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | settle => rfl
  | _ => simp [Instr.kind?] at h

theorem kind_barrier (h : x.kind? = some .barrier) : x = .barrier := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | barrier => rfl
  | _ => simp [Instr.kind?] at h

theorem kind_bfence (h : x.kind? = some .bfence) : x = .wgmmaFence := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | wgmmaFence => rfl
  | _ => simp [Instr.kind?] at h

theorem kind_bcommit (h : x.kind? = some .bcommit) : x = .wgmmaCommit := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | wgmmaCommit => rfl
  | _ => simp [Instr.kind?] at h

theorem kind_bwait (h : x.kind? = some .bwait) : ∃ n, x = .wgmmaWait n := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | wgmmaWait n => exact ⟨n, rfl⟩
  | _ => simp [Instr.kind?] at h

theorem kind_expect (h : x.kind? = some .expect) : ∃ b n, x = .expectTx b n := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | expectTx b n => exact ⟨b, n, rfl⟩
  | _ => simp [Instr.kind?] at h

theorem kind_fwait (h : x.kind? = some .fwait) : ∃ b p, x = .mbarWait (.full b) p := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h; exact ⟨_, _, rfl⟩
  | _ => simp [Instr.kind?] at h

theorem kind_ewait (h : x.kind? = some .ewait) : ∃ b p, x = .mbarWait (.empty b) p := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h; exact ⟨_, _, rfl⟩
  | _ => simp [Instr.kind?] at h

theorem kind_arrive (h : x.kind? = some .arrive) : ∃ b, x = .mbarArrive b := by
  cases x with
  | mbarWait b p => cases b <;> simp [Instr.kind?] at h
  | mbarArrive b => exact ⟨b, rfl⟩
  | _ => simp [Instr.kind?] at h

theorem getLast?_take (h : q < P.length) : (P.take (q + 1)).getLast? = P[q]? := by
  rw [List.getLast?_eq_getElem?]
  simp [List.length_take, Nat.min_eq_left (by omega : q + 1 ≤ P.length)]

theorem Obeys.settleAt (o : Obeys R P) :
    ∃ qs, nth .settle P 0 = some qs ∧ P[qs]? = some .settle ∧ P.length = qs + 1 := by
  obtain ⟨qs, hs, -⟩ := nth_lt (k := .settle) (n := 0) (p := P.length) (P := P)
    (by rw [List.take_length, o.settled]; omega)
  obtain ⟨x, hx, kind⟩ := nth_instr hs
  rw [kind_settle kind] at hx
  refine ⟨qs, hs, hx, ?_⟩
  have lt := len_of hx
  apply Classical.byContradiction
  intro ne
  have hy : P[qs + 1]? = some P[qs + 1] := List.getElem?_eq_getElem (by omega)
  have fresh := o.fresh _ _ hy
  have := counts_at hx .settle
  have := counts_take_mono P .settle (Nat.le_refl (qs + 1))
  simp [Instr.kind?] at *
  omega

theorem Obeys.total (o : Obeys R P) (h : nth .settle P 0 = some qs) (k : Kind) :
    counts P k = counts (P.take qs) k + if k = .settle then 1 else 0 := by
  obtain ⟨qs', hs, hx, len⟩ := o.settleAt
  rw [hs] at h; cases h
  have : P = P.take (qs + 1) := by rw [List.take_of_length_le (by omega)]
  conv => lhs; rw [this]
  rw [counts_at hx]
  by_cases e : k = .settle
  · subst e; simp [Instr.kind?]
  · simp [Instr.kind?, Ne.symm e, e]

theorem Obeys.beforeSettle (o : Obeys R P) (h : nth .settle P 0 = some qs) (hq : P[q]? = some x)
    (ne : x ≠ .settle) : q < qs := by
  have f := o.fresh q x hq
  have le := le_nth h (p := q) (by omega)
  rcases Nat.eq_or_lt_of_le le with rfl | lt
  · obtain ⟨y, hy, ky⟩ := nth_instr h
    rw [hq] at hy; cases hy
    exact (ne (kind_settle ky)).elim
  · exact lt

theorem Obeys.next (o : Obeys R P) (h : P[q]? = some (.expectTx b n)) :
    ∃ s b' src n', P[q + 1]? = some (.tmaLoad s b' src n') := by
  obtain ⟨qs, hs, hx, len⟩ := o.settleAt
  have lt := o.beforeSettle hs h (by simp)
  have hy : P[q + 1]? = some P[q + 1] := List.getElem?_eq_getElem (by omega)
  have l := o.loads _ _ hy
  rw [getLast?_take (by omega), h] at l
  revert l hy
  cases P[q + 1] <;> simp [afterExpect, isLoad]
  intro h; exact ⟨_, _, _, _, h⟩

/-! ### Timelines of one role -/

/-- A role has executed the first `done` instructions of its program, instruction `q` at `time q`. -/
structure Track where
  done : Nat
  time : Nat → Nat

def Track.at (r : Track) : Option Nat → Event
  | some q => event r.done q (r.time q)
  | none => none

structure Track.Mono (r : Track) : Prop where
  order : ∀ p q, p < q → q < r.done → r.time p < r.time q

theorem Track.Mono.lt (m : Track.Mono r) (h : p < q) : r.at (some p) ≺ r.at (some q) := by
  simp only [Track.at, event_edge]
  intro hq
  exact ⟨by omega, m.order p q h hq⟩

theorem Track.Mono.later (m : Track.Mono r) (ab : a ≺ r.at (some p)) (h : p ≤ q) : a ≺ r.at (some q) := by
  rcases Nat.eq_or_lt_of_le h with rfl | lt
  · exact ab
  · exact ab.trans (m.lt lt)

theorem Track.Mono.ordered (m : Track.Mono r) (h : ∀ q, o' = some q → ∃ p, o = some p ∧ p < q) :
    r.at o ≺ r.at o' := by
  cases o' with
  | none => intro t ht; simp [Track.at] at ht
  | some q => obtain ⟨p, rfl, lt⟩ := h q rfl; exact m.lt lt

theorem edge_none_left {b : Event} (h : (none : Event) ≺ b) : b = none := by
  cases b with
  | none => rfl
  | some t => obtain ⟨u, hu, _⟩ := h t rfl; cases hu

theorem edge_none_right {a : Event} : a ≺ (none : Event) := by intro t ht; cases ht

/-! ### Operands of numbered instructions -/

def operand (P : List Instr) (k : Kind) (n : Nat) : Option Instr := (nth k P n).bind (P[·]?)

def Bar.slot : Bar → Nat
  | .full s | .empty s => s

/-- The barrier slot of the `n`-th instruction of kind `k`. -/
def barOf (P : List Instr) (k : Kind) (n : Nat) : Option Nat :=
  match operand P k n with
  | some (.expectTx b _) | some (.mbarWait b _) | some (.mbarArrive b) => some b.slot
  | _ => none

/-- The full barrier of fill `j`, from its `expect_tx`. -/
def fillBar (P : List Instr) (j : Nat) : Nat := (barOf P .expect j).getD 0

/-- The first `tma_load` of fill `j` follows its `expect_tx`. -/
def loadOf (P : List Instr) (j : Nat) : Option Instr := (nth .expect P j).bind (fun q => P[q + 1]?)

def loadSlot (P : List Instr) (j : Nat) : Option Nat :=
  match loadOf P j with
  | some (.tmaLoad s _ _ _) => some s
  | _ => none

def loadSrc (P : List Instr) (j : Nat) : Option Nat :=
  match loadOf P j with
  | some (.tmaLoad _ _ src _) => some src
  | _ => none

theorem barOf_eq (h : nth k P n = some q) (hq : P[q]? = some x) :
    barOf P k n = match x with
      | .expectTx b _ | .mbarWait b _ | .mbarArrive b => some b.slot
      | _ => none := by
  simp only [barOf, operand, h, Option.bind_some, hq]
  cases x <;> rfl

theorem barOf_none (h : nth k P n = none) : barOf P k n = none := by
  simp [barOf, operand, h]

theorem barOf_wait (h : nth k P n = some q) (hq : P[q]? = some (.mbarWait b p)) : barOf P k n = some b.slot := by
  rw [barOf_eq h hq]

theorem barOf_expectTx (h : nth k P n = some q) (hq : P[q]? = some (.expectTx b m)) : barOf P k n = some b.slot := by
  rw [barOf_eq h hq]

theorem barOf_arrive (h : nth k P n = some q) (hq : P[q]? = some (.mbarArrive b)) : barOf P k n = some b.slot := by
  rw [barOf_eq h hq]

theorem loadOf_none (h : nth .expect P j = none) : loadOf P j = none := by
  simp [loadOf, h]

theorem loadOf_some (h : nth .expect P j = some q) : loadOf P j = P[q + 1]? := by
  simp [loadOf, h]

/-! ### Bytes -/

/-- Bytes still owed to the open fill; `none` outside a fill. -/
def owe : Option Nat → Instr → Option (Option Nat)
  | o, .expectTx _ n => if o.getD 0 = 0 then some (some n) else none
  | some o, .tmaLoad _ _ _ n => if n ≤ o then some (some (o - n)) else none
  | none, .tmaLoad .. => none
  | o, _ => if o.getD 0 = 0 then some none else none

def owes : Option Nat → List Instr → Option (Option Nat)
  | o, [] => some o
  | o, x :: xs => (owe o x).bind (owes · xs)

/-- Each `expect_tx` announces exactly the bytes of the `tma_load`s that follow it. -/
def bytesMatch (P : List Instr) : Bool :=
  match owes none P with
  | some o => o.getD 0 == 0
  | none => false

theorem owes_append : owes o (A ++ B) = (owes o A).bind (owes · B) := by
  induction A generalizing o with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.cons_append, owes]
    cases owe o x <;> simp [ih]

theorem owes_loads (B : Nat) (n : Nat) :
    owes (some (n * B)) (List.replicate n (.tmaLoad s b src B)) = some (some 0) := by
  induction n with
  | zero => simp [owes]
  | succ n ih =>
    have hb : B ≤ (n + 1) * B := by rw [Nat.succ_mul]; omega
    simp only [List.replicate_succ, owes, owe, hb, ite_true, Option.bind_some]
    rw [show (n + 1) * B - B = n * B by rw [Nat.succ_mul]; omega]
    exact ih

theorem owes_other (h : x.kind? ≠ some .expect) (hl : isLoad x = false) (o : Nat) (zero : o = 0) :
    owe (some o) x = some none := by
  subst zero
  cases x <;> simp_all [owe, Instr.kind?, isLoad]

def Clear (o : Option Nat) : Prop := o.getD 0 = 0

/-- A segment that leaves no fill owing bytes. -/
def Balanced (L : List Instr) : Prop := ∀ o, Clear o → ∃ o', owes o L = some o' ∧ Clear o'

theorem Balanced.append (a : Balanced A) (b : Balanced B) : Balanced (A ++ B) := by
  intro o ho
  obtain ⟨o', h, c⟩ := a o ho
  obtain ⟨o'', h', c'⟩ := b o' c
  exact ⟨o'', by rw [owes_append, h]; exact h', c'⟩

theorem Balanced.nil : Balanced [] := fun o ho => ⟨o, rfl, ho⟩

theorem Balanced.plain (h : x.kind? ≠ some .expect) (hl : isLoad x = false) : Balanced [x] := by
  intro o ho
  refine ⟨none, ?_, rfl⟩
  simp only [Clear] at ho
  cases x <;> simp_all [owes, owe, Instr.kind?, isLoad]

theorem Balanced.replicate (b : Balanced [x]) : Balanced (List.replicate n x) := by
  induction n with
  | zero => exact Balanced.nil
  | succ n ih => rw [List.replicate_succ, ← List.singleton_append]; exact b.append ih

theorem Balanced.fill : Balanced (.expectTx b (n * B) :: List.replicate n (.tmaLoad s b' src B)) := by
  intro o ho
  refine ⟨some 0, ?_, rfl⟩
  simp only [Clear] at ho
  simp only [owes, owe, ho, ite_true, Option.bind_some]
  exact owes_loads B n

theorem bytesMatch_of (b : Balanced P) : bytesMatch P = true := by
  obtain ⟨o, h, c⟩ := b none rfl
  simp only [Clear] at c
  simp [bytesMatch, h, c]

/-! ### Diagnostics -/

/-- The first violated rule and its instruction index; index `P.length` refers to the end. -/
def diagnoseFrom (R : Rules) : Counts → Option Instr → Nat → List Instr → Option (Nat × String)
  | s, _, i, [] => if s .settle == 1 then none else some (i, "expected exactly one settle at the end")
  | s, prev, i, x :: xs => match (checks R s prev x).find? (!·.2) with
    | some (msg, _) => some (i, msg)
    | none => diagnoseFrom R (s.add x) (some x) (i + 1) xs

def diagnose (R : Rules) (P : List Instr) : Option (Nat × String) :=
  diagnoseFrom R (counts []) none 0 P

private theorem diagnoseFrom_none (h : diagnoseFrom R s prev i P = none) :
    ∃ r, walk R s prev P = some r ∧ r.1 .settle = 1 := by
  induction P generalizing s prev i with
  | nil => simp [diagnoseFrom] at h; exact ⟨_, rfl, h⟩
  | cons x xs ih =>
    simp only [diagnoseFrom] at h
    split at h
    · cases h
    · rename_i none_
      have ok : obeys R s prev x = true := by
        simp only [obeys, List.all_eq_true]
        intro r hr
        have := List.find?_eq_none.mp none_ r hr
        simpa using this
      simp only [walk, ok, ite_true]
      exact ih h

theorem diagnose_none (h : diagnose R P = none) : run R P = true := by
  obtain ⟨⟨s, p⟩, hw, hs⟩ := diagnoseFrom_none h
  simp only at hs
  simp [run, hw, hs]

/-! ### Text format -/

def parseBar : List String → Option (Bar × List String)
  | "full" :: b :: rest => do pure (.full (← b.toNat?), rest)
  | "empty" :: b :: rest => do pure (.empty (← b.toNat?), rest)
  | _ => none

def parse (ws : List String) : Option Instr :=
  match ws with
  | "init" :: rest => do
    let (b, rest) ← parseBar rest
    let [n] := rest | none
    pure (.init b (← n.toNat?))
  | "expect_tx" :: rest => do
    let (b, rest) ← parseBar rest
    let [n] := rest | none
    pure (.expectTx b (← n.toNat?))
  | "tma_load" :: s :: rest => do
    let (b, rest) ← parseBar rest
    let [src, n] := rest | none
    pure (.tmaLoad (← s.toNat?) b (← src.toNat?) (← n.toNat?))
  | "mbar_wait" :: rest => do
    let (b, rest) ← parseBar rest
    let [p] := rest | none
    pure (.mbarWait b (← p.toNat?))
  | "mbar_arrive" :: rest => do
    let (b, rest) ← parseBar rest
    let [] := rest | none
    pure (.mbarArrive b)
  | ["bar"] => some .barrier
  | ["wgmma_fence"] => some .wgmmaFence
  | ["read", b, s] => do pure (.read (← b.toNat?) (← s.toNat?))
  | ["wgmma_commit"] => some .wgmmaCommit
  | ["wgmma_wait", n] => do pure (.wgmmaWait (← n.toNat?))
  | ["settle"] => some .settle
  | _ => none

def Bar.render : Bar → String
  | .full s => s!"full {s}"
  | .empty s => s!"empty {s}"

def render : Instr → String
  | .init b n => s!"init {b.render} {n}"
  | .expectTx b n => s!"expect_tx {b.render} {n}"
  | .tmaLoad s b src n => s!"tma_load {s} {b.render} {src} {n}"
  | .mbarWait b p => s!"mbar_wait {b.render} {p}"
  | .mbarArrive b => s!"mbar_arrive {b.render}"
  | .barrier => "bar"
  | .wgmmaFence => "wgmma_fence"
  | .read b s => s!"read {b} {s}"
  | .wgmmaCommit => "wgmma_commit"
  | .wgmmaWait n => s!"wgmma_wait {n}"
  | .settle => "settle"

/-- A role header and the instructions after it, with their line numbers and text. -/
structure Section where
  header : List String
  line : Nat
  ops : Array (Nat × String × Instr)

/-- Splits the input at `role` lines; `#` starts a comment. -/
def sections (text : String) : Except String (Array Section) := Id.run do
  let mut out : Array Section := #[]
  let mut line := 0
  for raw in text.splitOn "\n" do
    line := line + 1
    let body := ((raw.splitOn "#").headD "").trimAscii.toString
    if body.isEmpty then continue
    let ws := (body.splitOn " ").filter (· ≠ "")
    if ws.head? = some "role" then
      out := out.push ⟨ws.tail, line, #[]⟩
      continue
    let some op := parse ws | return .error s!"line {line}: cannot parse '{body}'"
    let some cur := out.back? | return .error s!"line {line}: instruction before the first role header"
    out := out.pop.push { cur with ops := cur.ops.push (line, body, op) }
  return .ok out

end Ring.Tma
