import Ring.Timeline

namespace Ring.Async
open Timeline

/-- One thread's stream of committed copy groups. Ticket numbers start at zero.
    The compiler commits even when a predicated thread copied zero bytes. -/
structure Stream where
  slot : Nat → Nat
  sourceBlock : Nat → Nat
  issue : Nat → Event
  commit : Nat → Event
  complete : Nat → Event
  waitGroup : Nat → Event     -- dynamic invocation at loop iteration i
  proxyFence : Nat → Event
  settle : Event

/-- A read of shared memory returns with the load, or remains live until
    completion of its warpgroup batch. Several reads may name the same batch. -/
inductive Reader where
  | load
  | wgmma (batch : Nat)

structure Read (threads : Nat) where
  lane : Fin threads
  block : Nat
  slot : Nat
  issue : Event
  reader : Reader

structure CTA (threads : Nat) where
  writer : Fin threads → Stream
  fence : Nat → Event         -- release of bar.sync in refilled
  warpgroup : Fin threads → Nat
  batchFence : Nat → Nat → Event
  batchCommit : Nat → Nat → Event
  batchDone : Nat → Nat → Event
  batchWait : Nat → Nat → Event   -- wait_group 0 after this batch

def retired (c : CTA threads) (r : Read threads) : Event :=
  match r.reader with
  | .load => r.issue
  | .wgmma b => c.batchDone (c.warpgroup r.lane) b

/-- exact=false. The last real block supplies the trailing fills; each still
    has its own ticket, physical destination and commit group. -/
def source (d count j : Nat) := if j < d - 1 then j else min j (count - 1)

def writes (c : CTA threads) (d count : Nat) (lane : Fin threads) : Writes :=
  ⟨d - 1 + count, (c.writer lane).slot, (c.writer lane).sourceBlock, (c.writer lane).issue, (c.writer lane).complete⟩

/-- Ordering constraints on a finite prefix of the generated lag-zero fold.
    A fence timestamp denotes collective release. Device events have no
    cross-thread ordering except what program order and the barrier imply. -/
structure Fold (c : CTA threads) (d count pending : Nat) (proxy : Bool) : Prop where
  fillSlot : ∀ l j, j < d - 1 + count → (c.writer l).slot j = j % d
  fillSource : ∀ l j, j < d - 1 + count → (c.writer l).sourceBlock j = source d count j
  issueCommit : ∀ l j, (c.writer l).issue j ≺ (c.writer l).commit j
  commitDone : ∀ l j, (c.writer l).commit j ≺ (c.writer l).complete j
  commitOrder : ∀ l j, j + 1 < d - 1 + count →
    (c.writer l).commit j ≺ (c.writer l).issue (j+1)
  -- Exactly d-1+i groups committed before this thread's wait. The
  -- wait does not test arrivals, bytes, or another thread's copy groups.
  priorCommit : ∀ l i, i < count → (c.writer l).commit (d - 2 + i) ≺ (c.writer l).waitGroup i
  waitOld : ∀ l i j, i < count → j + pending < d - 1 + i →
    (c.writer l).complete j ≺ (c.writer l).waitGroup i
  publish : ∀ l i, i < count → proxy = true →
    (c.writer l).waitGroup i ≺ (c.writer l).proxyFence i
  arrive : ∀ l i, i < count →
    (if proxy then (c.writer l).proxyFence i else (c.writer l).waitGroup i) ≺ c.fence i
  refill : ∀ l i, i < count → c.fence i ≺ (c.writer l).issue (i + (d - 1))
  -- async_wait at exit drains this issuing thread, including clamped tail copies.
  drain : ∀ l j, j < d - 1 + count → (c.writer l).complete j ≺ (c.writer l).settle

/-- The host instruction order for one observed read. All lanes of one warpgroup name the same batch events. The next wait is present only on a non-final iteration. -/
structure ReadCode (c : CTA threads) (d count : Nat) (proxy : Bool) (r : Read threads) : Prop where
  bound : r.block < count
  operand : r.slot = r.block % d
  afterRefill : (c.writer r.lane).commit (r.block + (d - 1)) ≺ r.issue
  proxyAccess : ∀ b, r.reader = .wgmma b → proxy = true ∧ c.batchFence (c.warpgroup r.lane) b ≺ r.issue
  stepOrder : match r.reader with
    | .load => r.block + 1 < count → r.issue ≺ (c.writer r.lane).waitGroup (r.block+1)
    | .wgmma b =>
      r.issue ≺ c.batchCommit (c.warpgroup r.lane) b ∧ c.batchCommit (c.warpgroup r.lane) b ≺ c.batchDone (c.warpgroup r.lane) b ∧
      c.batchDone (c.warpgroup r.lane) b ≺ c.batchWait (c.warpgroup r.lane) b ∧
      (r.block + 1 < count → c.batchWait (c.warpgroup r.lane) b ≺ (c.writer r.lane).waitGroup (r.block+1))

theorem Fold.waitFence (f : Fold c d count n proxy) (l) (hi : i < count) :
    (c.writer l).waitGroup i ≺ c.fence i := by
  have h := f.arrive l i hi
  cases proxy with
  | false => exact h
  | true => exact (f.publish l i hi rfl).trans h

theorem Fold.fenceNext {c : CTA threads} (f : Fold c d count n proxy) (l : Fin threads) (hd : 2 ≤ d) (hi : i+1 < count) :
    c.fence i ≺ c.fence (i+1) := by
  have eq : d - 2 + (i+1) = i + (d-1) := by omega
  have prior := f.priorCommit l (i+1) hi
  rw [eq] at prior
  exact ((f.refill l i (by omega)).trans (f.issueCommit l _)).trans
    (prior.trans (f.waitFence l hi))

private theorem ReadCode.nextWait (h : ReadCode c d count proxy r) (bound : r.block+1 < count) :
    retired c r ≺ (c.writer r.lane).waitGroup (r.block+1) := by
  have order := h.stepOrder
  cases eq : r.reader with
  | load => simp [eq, retired] at order ⊢; exact order bound
  | wgmma b =>
    simp [eq, retired] at order ⊢
    exact order.2.2.1.trans (order.2.2.2 bound)

/-- T2 for every copy issuer and every reader (including different threads).
    The release of the CTA fence in the refill path joins their local orders. -/
theorem overwrite_after_last_read (f : Fold c d count n proxy) (hd : 2 ≤ d)
    (h : ReadCode c d count proxy r) (l) (bound : k < d - 1 + count)
    (later : r.block < k) (same : (c.writer l).slot k = r.slot) :
    retired c r ≺ (c.writer l).issue k := by
  rw [f.fillSlot l k bound] at same
  have gap := circle_gap (by omega : 0 < d) (h.operand.symm.trans same.symm) later
  let q := k - (d-1)
  have atFill : q + (d-1) = k := by dsimp [q]; omega
  have qb : q < count := by dsimp [q]; omega
  have reach : r.block + 1 ≤ q := by dsimp [q]; omega
  have first := (h.nextWait (by omega)).trans (f.waitFence r.lane (by omega))
  have toFence : retired c r ≺ c.fence q := by
    rcases Nat.eq_or_lt_of_le reach with eq | lt
    · rwa [← eq]
    · exact first.trans (chain (fun _ => f.fenceNext r.lane hd) lt qb)
  exact atFill ▸ toFence.trans (f.refill l q qb)

/-- T1 and stability throughout a read. At an issue cut, or at any later
    cut before retirement, every issuer's slice has the right source, ticket
    and physical slot, and no copy into it is in flight. -/
theorem read_current_generation (f : Fold c d count (d-2) proxy) (hd : 2 ≤ d)
    (h : ReadCode c d count proxy r) (l) (issued : By r.issue cut)
    (active : r.issue = some cut ∨ ¬ By (retired c r) cut) :
    (writes c d count l).Shows r.slot r.block r.block cut := by
  have bound := h.bound
  have fence := ((f.refill r.lane r.block bound).trans
    ((f.issueCommit r.lane _).trans h.afterRefill)).byTime issued
  have src : source d count r.block = r.block := by unfold source; split <;> omega
  have ticket : r.block < d - 1 + count := by omega
  apply Writes.shows_of_bounds _ ticket ((f.fillSlot l _ ticket).trans h.operand.symm)
    ((f.fillSource l _ ticket).trans src)
    ((f.issueCommit l _).trans (f.commitDone l _))
  · intro k le _
    exact ((f.waitOld l r.block k bound (by omega)).trans (f.waitFence l bound)).byTime fence
  · intro k later kb same started
    have reuse := overwrite_after_last_read f hd h l kb later same
    rcases active with atIssue | live
    · have order := h.stepOrder
      obtain ⟨t, ht, le⟩ := started
      obtain ⟨u, hu, lt⟩ := reuse _ ht
      cases eq : r.reader with
      | load => simp [retired, eq, atIssue] at hu; omega
      | wgmma b =>
        simp [eq, retired] at hu order
        obtain ⟨s, hs, _⟩ := (order.1.trans order.2.1) _ hu
        rw [atIssue] at hs
        cases hs
        omega
    · exact live (reuse.byTime started)

end Ring.Async
