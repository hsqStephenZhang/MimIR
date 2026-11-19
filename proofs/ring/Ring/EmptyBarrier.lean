import Ring.Phase

namespace Ring.EmptyBarrier
open Timeline

/-- Projection on one physical full/empty pair. Each of the eight warp
    leaders in two consumer warpgroups contributes a *separate* arrival. -/
structure Slot where
  acquire : Nat → Event
  start : Nat → Event
  publish : Nat → Event
  poll : Fin 2 → Nat → Event
  give : Fin 8 → Nat → Event

def group (w : Fin 8) : Fin 2 := ⟨w.val / 4, by have := w.isLt; omega⟩

def total : (n : Nat) → (Fin n → Nat) → Nat
  | 0, _ => 0
  | n+1, f => total n (fun i => f i.succ) + f 0

private theorem total_bounds (f : Fin n → Nat) (bounds : ∀ i, lo ≤ f i ∧ f i ≤ hi) :
    n*lo ≤ total n f ∧ total n f ≤ n*hi ∧ (total n f = n*hi → ∀ i, f i = hi) := by
  induction n with
  | zero => simp [total]
  | succ n ih =>
    obtain ⟨lower, upper, full⟩ := ih (fun i => f i.succ) (fun i => bounds i.succ)
    have first := bounds 0
    simp only [total, Nat.succ_mul]
    refine ⟨by omega, by omega, ?_⟩
    intro eq i
    exact Fin.cases (by omega) (full (by omega)) i

def credits (s : Slot) (n t : Nat) (w : Fin 8) :=
  tally (fun k => occurred (s.give w k) t) n

def emptyPhase (s : Slot) (n t : Nat) := total 8 (credits s n t) / 8
def fullPhase (s : Slot) (n t : Nat) := tally (fun k => occurred (s.publish k) t) n

structure Protocol (s : Slot) (n : Nat) : Prop where
  acquireStart : ∀ k, k < n → s.acquire k ≺ s.start k
  startPublish : ∀ k, k < n → s.start k ≺ s.publish k
  nextAcquire : ∀ k, k+1 < n → s.start k ≺ s.acquire (k+1)
  nextPoll : ∀ g k, k+1 < n → s.poll g k ≺ s.poll g (k+1)
  releaseOrder : ∀ w k, k < n → s.poll (group w) k ≺ s.give w k
  empty : ∀ k t, k < n → s.acquire k = some t → emptyPhase s n t % 2 ≠ (k+1)%2
  full : ∀ g k t, k < n → s.poll g k = some t → fullPhase s n t % 2 ≠ k%2

def Released (s : Slot) (k t : Nat) := ∀ w j, j < k → By (s.give w j) t

private theorem tally_true (k : Nat) : tally (fun _ => true) k = k := by
  induction k <;> simp_all [tally]

private theorem prefix_count {e : Nat → Event} (bound : k ≤ n)
    (hp : ∀ j, j < n → (By (e j) t ↔ j < k)) :
    tally (fun j => occurred (e j) t) n = k := by
  rw [tally_prefix (q := fun _ => true) (i := k), Nat.min_eq_right bound, tally_true]
  simpa [occurred_iff] using hp

theorem fullPhase_eq (bound : k ≤ n) (hp : ∀ j, j < n → (By (s.publish j) t ↔ j < k)) :
    fullPhase s n t = k := prefix_count bound hp

theorem emptyPhase_eq (bound : k ≤ n)
    (hp : ∀ w j, j < n → (By (s.give w j) t ↔ j < k)) : emptyPhase s n t = k := by
  have counts (w : Fin 8) : credits s n t w = k := prefix_count bound (hp w)
  obtain ⟨lo, hi, _⟩ := total_bounds (lo := k) (hi := k) (credits s n t) (fun w => by simp [counts])
  unfold emptyPhase
  omega

/-- Joint induction on event time: a reader cannot lap the producer and a
    producer cannot lap any of the eight warp leaders. Both guards inspect
    only parity, even on a partially paid empty phase. -/
theorem handshake (p : Protocol s n) (t : Nat) :
    (∀ g k, k < n → s.poll g k = some t → By (s.publish k) t) ∧
    (∀ k, k < n → s.acquire k = some t → Released s k t) := by
  classical
  induction t using Nat.strongRecOn with
  | ind t ih =>
    constructor
    · intro g k kb atPoll
      apply Classical.byContradiction
      intro missing
      apply p.full g k t kb atPoll
      apply congrArg (· % 2)
      apply prefix_count (by omega : k ≤ n)
      intro j jb
      constructor
      · intro published
        rcases Nat.lt_trichotomy j k with earlier | rfl | later
        · exact earlier
        · exact (missing published).elim
        · obtain ⟨u, hu, le⟩ := published
          obtain ⟨v, hv, vu⟩ := ((p.acquireStart j jb).trans (p.startPublish j jb)) u hu
          let w : Fin 8 := ⟨4*g.val, by have := g.isLt; omega⟩
          have same : group w = g := by apply Fin.ext; simp [group, w]
          obtain ⟨a, ha, av⟩ := (ih v (by omega)).2 j jb hv w k later
          obtain ⟨b, hb, ba⟩ := p.releaseOrder w k kb a ha
          rw [same, atPoll] at hb
          cases hb
          omega
      · intro earlier
        obtain ⟨u, hu, ut⟩ := chain (p.nextPoll g) earlier kb t atPoll
        exact ((ih u ut).1 g j jb hu).mono (by omega)
    · intro k kb atAcquire
      cases k with
      | zero => intro w j hj; omega
      | succ k =>
        obtain ⟨u, hu, ut⟩ := ((p.acquireStart k (by omega)).trans (p.nextAcquire k kb)) t atAcquire
        have old := (ih u ut).2 k (by omega) hu
        have future (w : Fin 8) (j) (jb : j < n) (later : k+1 ≤ j) : ¬ By (s.give w j) t := by
          rintro ⟨v, hv, vt⟩
          obtain ⟨a, ha, av⟩ := p.releaseOrder w j jb v hv
          obtain ⟨b, hb, ba⟩ := (ih a (by omega)).1 _ j jb ha
          obtain ⟨c, hc, cb⟩ := ((p.acquireStart j jb).trans (p.startPublish j jb)) b hb
          rcases Nat.eq_or_lt_of_le later with rfl | lt
          · rw [atAcquire] at hc; cases hc; omega
          · obtain ⟨x, hx, xc⟩ := chain
              (fun i hi => (p.acquireStart i (by omega)).trans (p.nextAcquire i hi)) lt jb c hc
            rw [atAcquire] at hx; cases hx; omega
        have paid (w : Fin 8) : credits s n t w = if By (s.give w k) t then k+1 else k := by
          unfold credits
          split
          · rename_i last
            apply prefix_count (by omega)
            intro j jb
            constructor
            · intro h; by_cases lt : j < k+1
              · exact lt
              · exact (future w j jb (by omega) h).elim
            · intro lt
              rcases Nat.eq_or_lt_of_le (by omega : j ≤ k) with rfl | lt
              · exact last
              · exact (old w j lt).mono (by omega)
          · rename_i last
            apply prefix_count (by omega)
            intro j jb
            constructor
            · intro h
              rcases Nat.lt_trichotomy j k with lt | rfl | lt
              · exact lt
              · exact (last h).elim
              · exact (future w j jb lt h).elim
            · intro lt; exact (old w j lt).mono (by omega)
        have bounds (w : Fin 8) : k ≤ credits s n t w ∧ credits s n t w ≤ k+1 := by
          rw [paid]; split <;> omega
        obtain ⟨low, high, allPaid⟩ := total_bounds _ bounds
        have parity := p.empty (k+1) t kb atAcquire
        unfold emptyPhase at parity
        have exactSum : total 8 (credits s n t) = 8*(k+1) := by omega
        intro w j lt
        rcases Nat.eq_or_lt_of_le (by omega : j ≤ k) with rfl | lt
        · have last := allPaid exactSum w
          rw [paid] at last
          split at last <;> first | assumption | omega
        · exact (old w j lt).mono (by omega)

end Ring.EmptyBarrier
