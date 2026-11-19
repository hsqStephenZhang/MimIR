import Ring.Phase

namespace Ring.Cooperative
open Timeline

structure CTA (groups : Nat) where
  leader : Fin groups
  start : Nat → Event
  arrived : Nat → Event
  bytesComplete : Nat → Event
  publish : Nat → Event
  fullBarrier : Nat → Nat
  fillSlot : Nat → Nat
  source : Nat → Nat
  pollSlot : Fin groups → Nat → Nat
  poll : Fin groups → Nat → Event
  batchFence : Fin groups → Nat → Event
  commit : Fin groups → Nat → Event
  done_ : Fin groups → Nat → Event
  waitGroup : Fin groups → Nat → Event
  fence : Nat → Event
  finalWait : Fin groups → Event

/-- Both the TMA destination and its completion barrier are physical operands. -/
def phase (c : CTA groups) (count s cut : Nat) : Nat :=
  phaseAt c.fullBarrier c.publish count s cut

def writes (c : CTA groups) (count : Nat) : Writes :=
  ⟨count, c.fillSlot, c.source, c.start, c.publish⟩

/-- Exact, late-refill fold. Each warpgroup executes one WGMMA commit per
    block. Device completion and parity are shared, batch retirement is local. -/
structure Fold (c : CTA groups) (d count : Nat) : Prop where
  primeBound : d - 1 ≤ count
  primeOrder : ∀ j, j+1 < d-1 → c.start j ≺ c.start (j+1)
  primed : ∀ j, j < d-1 → c.start j ≺ c.poll c.leader 0
  leaderNext : ∀ i, i + (d-1) < count → c.start (i + (d-1)) ≺ c.poll c.leader (i+1)
  operand : ∀ j, j < count → c.fillSlot j = j % d
  barrier : ∀ j, j < count → c.fullBarrier j = j % d
  source : ∀ j, j < count → c.source j = j
  awaited : ∀ g i, i < count → c.pollSlot g i = i % d
  startArrival : ∀ j, c.start j ≺ c.arrived j
  startBytes : ∀ j, c.start j ≺ c.bytesComplete j
  arrival : ∀ j, c.arrived j ≺ c.publish j
  bytes : ∀ j, c.bytesComplete j ≺ c.publish j
  parity : ∀ g i t, i < count → c.poll g i = some t →
    phase c count (c.pollSlot g i) t % 2 ≠ (i / d) % 2
  acquired : ∀ g i, c.poll g i ≺ c.batchFence g i
  batch : ∀ g i, c.batchFence g i ≺ c.commit g i
  committed : ∀ g i, c.commit g i ≺ c.done_ g i
  waited : ∀ g i, c.commit g i ≺ c.waitGroup g i
  -- wait_group 1 returns only after all but the newest local group retired.
  oldGroups : ∀ g i j, j + 1 < i + 1 → c.done_ g j ≺ c.waitGroup g i
  arriveCTA : ∀ g i, c.waitGroup g i ≺ c.fence i
  nextPoll : ∀ g i, i+1 < count → c.fence i ≺ c.poll g (i+1)
  refill : ∀ i, i + (d-1) < count → c.fence i ≺ c.start (i + (d-1))
  drain : ∀ g i, i < count → c.done_ g i ≺ c.finalWait g

structure Read (groups : Nat) where
  group : Fin groups
  block : Nat
  slot : Nat
  issue : Event

structure ReadCode (c : CTA groups) (d count : Nat) (r : Read groups) : Prop where
  bound : r.block < count
  operand : r.slot = r.block % d
  fenced : c.batchFence r.group r.block ≺ r.issue
  committed : r.issue ≺ c.commit r.group r.block

private theorem Fold.pollFence (f : Fold c d count) (g i) : c.poll g i ≺ c.fence i :=
  (f.acquired g i).trans ((f.batch g i).trans ((f.waited g i).trans (f.arriveCTA g i)))

private theorem Fold.pollOrder (f : Fold c d count) (g) (lt : i < j) (bound : j < count) :
    c.poll g i ≺ c.poll g j :=
  chain (fun k hk => (f.pollFence g k).trans (f.nextPoll g k hk)) lt bound

private theorem Fold.pollBeforeReuse (f : Fold c d count) (g) (hd : 2 ≤ d)
    (gap : i+d ≤ k) (kb : k < count) : c.poll g i ≺ c.start k := by
  let q := k - (d-1)
  have eq : q + (d-1) = k := by dsimp [q]; omega
  have toPoll := f.pollOrder g (i := i) (j := q) (by dsimp [q]; omega) (by dsimp [q]; omega)
  exact eq ▸ (toPoll.trans (f.pollFence g q)).trans (f.refill q (by omega))

/-- Passing a one-bit poll identifies the publication of the actual ticket.
    No "poll implies ready" assumption is made. -/
theorem poll_published (f : Fold c d count) (hd : 2 ≤ d) (g) (bound : i < count)
    (polled : c.poll g i = some t) : By (c.publish i) t := by
  induction i using Nat.strongRecOn generalizing t with
  | ind i ih =>
    apply Classical.byContradiction
    intro missing
    have parity := f.parity g i t bound polled
    rw [f.awaited g i bound] at parity
    apply parity
    apply congrArg (· % 2)
    apply phase_before (by omega) (by omega : i ≤ count)
    intro k kb
    rw [f.barrier k kb]
    constructor
    · rintro ⟨same, published⟩
      refine ⟨?_, same⟩
      rcases Nat.lt_trichotomy k i with lt | rfl | later
      · exact lt
      · exact (missing published).elim
      · have gap := circle_gap (by omega : 0 < d) same.symm later
        have edge := (f.pollBeforeReuse g hd gap kb).trans ((f.startArrival k).trans (f.arrival k))
        obtain ⟨u, hu, le⟩ := published
        obtain ⟨v, hv, lt'⟩ := edge u hu
        rw [polled] at hv
        cases hv
        omega
    · rintro ⟨earlier, same⟩
      obtain ⟨u, hu, ut⟩ := f.pollOrder g earlier bound t polled
      exact ⟨same, (ih k earlier kb hu).mono (by omega)⟩

/-- T2 applies to every warpgroup, including one lagging behind the leader. -/
theorem overwrite_after_last_read (f : Fold c d count) (hd : 2 ≤ d)
    (h : ReadCode c d count r) (kb : k < count) (later : r.block < k)
    (same : c.fillSlot k = r.slot) : c.done_ r.group r.block ≺ c.start k := by
  rw [f.operand k kb, h.operand] at same
  have gap := circle_gap (by omega : 0 < d) same.symm later
  let q := k - (d-1)
  have eq : q + (d-1) = k := by dsimp [q]; omega
  have retired := f.oldGroups r.group q r.block (by dsimp [q]; omega)
  exact eq ▸ (retired.trans (f.arriveCTA r.group q)).trans (f.refill q (by omega))

/-- T1 at issue and stability up to the completion of this group's batch. -/
theorem read_current_generation (f : Fold c d count) (hd : 2 ≤ d)
    (h : ReadCode c d count r) (issued : By r.issue cut)
    (active : r.issue = some cut ∨ ¬ By (c.done_ r.group r.block) cut) :
    (writes c count).Shows r.slot r.block r.block cut := by
  have bound := h.bound
  obtain ⟨t, polled, le⟩ := ((f.acquired r.group r.block).trans h.fenced).byTime issued
  apply Writes.shows_of_bounds _ bound ((f.operand _ bound).trans h.operand.symm)
    (f.source _ bound) ((f.startArrival _).trans (f.arrival _))
  · intro k older _
    rcases Nat.eq_or_lt_of_le older with rfl | earlier
    · exact (poll_published f hd _ bound polled).mono le
    · obtain ⟨u, hu, _⟩ := f.pollOrder r.group earlier bound t polled
      exact (poll_published f hd _ (by omega) hu).mono (by omega)
  · intro k later kb same started
    have reuse := overwrite_after_last_read f hd h kb later same
    rcases active with atIssue | live
    · obtain ⟨u, hu, _⟩ := started
      obtain ⟨v, hv, _⟩ := (h.committed.trans ((f.committed _ _).trans reuse)) u hu
      rw [atIssue] at hv
      cases hv
      omega
    · exact live (reuse.byTime started)

end Ring.Cooperative
