import Ring.EmptyBarrier

namespace Ring.WarpSpecialized
open Timeline EmptyBarrier

/-- Number of real fills of a physical slot in a nonpersistent tile. -/
def turns (d count s : Nat) := if s < count then (count-1-s)/d + 1 else 0

theorem inTurns (hd : 0 < d) : k < turns d count s ↔ s+d*k < count := by
  unfold turns
  split
  · rw [Nat.lt_succ_iff, Nat.le_div_iff_mul_le hd, Nat.mul_comm k d]
    omega
  · omega

structure CTA where
  slots : Nat → Slot
  emptySlot : Nat → Nat
  panel : Nat → Nat
  fullSlot : Nat → Nat
  pollSlot : Fin 2 → Nat → Nat
  giveSlot : Fin 8 → Nat → Nat
  source : Nat → Nat
  arrived : Nat → Event
  bytesComplete : Nat → Event
  batchFence : Fin 2 → Nat → Event
  commit : Fin 2 → Nat → Event
  done_ : Fin 2 → Nat → Event
  waitGroup : Fin 2 → Nat → Event
  finalWait : Fin 2 → Event

def CTA.acquire (c : CTA) (d j : Nat) := (c.slots (c.emptySlot j)).acquire (j/d)
def CTA.start (c : CTA) (d j : Nat) := (c.slots (c.panel j)).start (j/d)
def CTA.publish (c : CTA) (d j : Nat) := (c.slots (c.fullSlot j)).publish (j/d)
def CTA.poll (c : CTA) (d : Nat) (g : Fin 2) (j : Nat) := (c.slots (c.pollSlot g j)).poll g (j/d)
def CTA.give (c : CTA) (d : Nat) (w : Fin 8) (j : Nat) := (c.slots (c.giveSlot w j)).give w (j/d)

def writes (c : CTA) (d count : Nat) : Writes :=
  ⟨count, c.panel, c.source, c.start d, c.publish d⟩

/-- Single CTA, one producer warp, two consumer warpgroups, lag one.
    Each consumer waits locally; there is no CTA rendezvous inside the loop. -/
structure Fold (c : CTA) (d count : Nat) : Prop where
  emptyOperand : ∀ j, j < count → c.emptySlot j = j%d
  destination : ∀ j, j < count → c.panel j = j%d
  fullOperand : ∀ j, j < count → c.fullSlot j = j%d
  polled : ∀ g j, j < count → c.pollSlot g j = j%d
  released : ∀ w j, j < count → c.giveSlot w j = j%d
  source : ∀ j, j < count → c.source j = j
  acquired : ∀ j, c.acquire d j ≺ c.start d j
  nextFill : ∀ j, j+1 < count → c.start d j ≺ c.acquire d (j+1)
  startArrival : ∀ j, c.start d j ≺ c.arrived j
  startBytes : ∀ j, c.start d j ≺ c.bytesComplete j
  arrival : ∀ j, c.arrived j ≺ c.publish d j
  bytes : ∀ j, c.bytesComplete j ≺ c.publish d j
  full : ∀ g j t, j < count → c.poll d g j = some t →
    let s := c.pollSlot g j
    fullPhase (c.slots s) (turns d count s) t % 2 ≠ (j/d)%2
  empty : ∀ j t, j < count → c.acquire d j = some t →
    let s := c.emptySlot j
    emptyPhase (c.slots s) (turns d count s) t % 2 ≠ (j/d+1)%2
  acquiredFull : ∀ g j, c.poll d g j ≺ c.batchFence g j
  fenced : ∀ g j, c.batchFence g j ≺ c.commit g j
  committed : ∀ g j, c.commit g j ≺ c.done_ g j
  waited : ∀ g j, c.commit g j ≺ c.waitGroup g j
  nextStep : ∀ g j, j+1 < count → c.waitGroup g j ≺ c.poll d g (j+1)
  oldGroups : ∀ g i j, j+1 < i+1 → c.done_ g j ≺ c.waitGroup g i
  drain : ∀ g j, j < count → c.done_ g j ≺ c.finalWait g
  givePrevious : ∀ w j, j+1 < count → c.waitGroup (group w) (j+1) ≺ c.give d w j
  giveLast : ∀ w j, j+1 = count → c.finalWait (group w) ≺ c.give d w j
  afterGive : ∀ w j, j+2 < count → c.give d w j ≺ c.poll d (group w) (j+2)

theorem Fold.retiredBeforeGive (f : Fold c d count) (w) (jb : j < count) :
    c.done_ (group w) j ≺ c.give d w j := by
  by_cases more : j+1 < count
  · exact (f.oldGroups _ (j+1) j (by omega)).trans (f.givePrevious w j more)
  · exact (f.drain _ j jb).trans (f.giveLast w j (by omega))

private theorem Fold.pollOrder (f : Fold c d count) (g) (hi : i < j) (hj : j < count) :
    c.poll d g i ≺ c.poll d g j :=
  chain (fun k kb => (f.acquiredFull g k).trans ((f.fenced g k).trans
    ((f.waited g k).trans (f.nextStep g k kb)))) hi hj

/-- Demultiplex the two independent instruction streams using their certified
    panel, wait-barrier and release-barrier operands. -/
theorem Fold.slotProtocol (f : Fold c d count) (hd : 0 < d) (hs : s < d) :
    Protocol (c.slots s) (turns d count s) := by
  let j := fun h => s+d*h
  have mod (h) : j h % d = s := by simp [j, Nat.mod_eq_of_lt hs]
  have div (h) : j h / d = h := by simp [j, Nat.add_mul_div_left _ _ hd, Nat.div_eq_of_lt hs]
  -- The unbounded fields of Slot are observed only at real visits.
  -- The order constructors below are used only on that finite domain.
  have bound (h) (hb : h < turns d count s) : j h < count := (inTurns hd).mp hb
  have acquire (h) (hb : j h < count) : c.acquire d (j h) = (c.slots s).acquire h := by
    simp [CTA.acquire, f.emptyOperand _ hb, mod, div]
  have start (h) (hb : j h < count) : c.start d (j h) = (c.slots s).start h := by
    simp [CTA.start, f.destination _ hb, mod, div]
  have pub_ (h) (hb : j h < count) : c.publish d (j h) = (c.slots s).publish h := by
    simp [CTA.publish, f.fullOperand _ hb, mod, div]
  have poll (g h) (hb : j h < count) : c.poll d g (j h) = (c.slots s).poll g h := by
    simp [CTA.poll, f.polled _ _ hb, mod, div]
  have give (w h) (hb : j h < count) : c.give d w (j h) = (c.slots s).give w h := by
    simp [CTA.give, f.released _ _ hb, mod, div]
  have next (h) : j (h+1) = j h + d := by simp [j, Nat.mul_succ, Nat.add_assoc]
  constructor
  · intro h hb
    rw [← acquire h (bound _ hb), ← start h (bound _ hb)]
    exact f.acquired _
  · intro h hb
    rw [← pub_ h (bound _ hb), ← start h (bound _ hb)]
    exact (f.startArrival _).trans (f.arrival _)
  · intro h hb
    have later := bound _ hb
    have earlier := bound h (by omega)
    rw [← start h earlier, ← acquire _ later]
    rw [next] at later ⊢
    have first := f.nextFill (j h) (by omega)
    by_cases one : d = 1
    · simpa [one] using first
    · exact first.trans (chain (fun k kb => (f.acquired k).trans (f.nextFill k kb)) (by omega) later)
  · intro g h hb
    rw [← poll g h (bound _ (by omega)), ← poll g _ (bound _ hb)]
    exact f.pollOrder g (by rw [next]; omega) (bound _ hb)
  · intro w h hb
    have jb := bound _ hb
    rw [← give w h jb, ← poll (group w) h jb]
    exact (f.acquiredFull _ _).trans ((f.fenced _ _).trans
      ((f.committed _ _).trans (f.retiredBeforeGive w jb)))
  · intro h t hb ha
    have jb := bound _ hb
    rw [← acquire _ jb] at ha
    simpa [f.emptyOperand _ jb, mod, div] using f.empty _ t jb ha
  · intro g h t hb hp
    have jb := bound _ hb
    rw [← poll _ _ jb] at hp
    simpa [f.polled _ _ jb, mod, div] using f.full g _ t jb hp

structure Read where
  group : Fin 2
  block : Nat
  slot : Nat
  issue : Event

structure ReadCode (c : CTA) (d count : Nat) (r : Read) : Prop where
  bound : r.block < count
  operand : r.slot = r.block%d
  fenced : c.batchFence r.group r.block ≺ r.issue
  committed : r.issue ≺ c.commit r.group r.block

theorem poll_published (f : Fold c d count) (hd : 0 < d) (g) (jb : j < count)
    (polled : c.poll d g j = some t) : By (c.publish d j) t := by
  have hb : j/d < turns d count (j%d) := (inTurns hd).mpr (by simpa [Nat.mod_add_div] using jb)
  rw [CTA.poll, f.polled _ _ jb] at polled
  rw [CTA.publish, f.fullOperand _ jb]
  exact (handshake (f.slotProtocol hd (Nat.mod_lt _ hd)) t).1 g _ hb polled

theorem overwrite_after_last_read (f : Fold c d count) (hd : 0 < d) (h : ReadCode c d count r)
    (kb : k < count) (later : r.block < k) (same : c.panel k = r.slot) :
    c.done_ r.group r.block ≺ c.start d k := by
  rw [f.destination _ kb, h.operand] at same
  have kb' : k/d < turns d count (k%d) := (inTurns hd).mpr (by simpa [Nat.mod_add_div] using kb)
  have earlier : r.block/d < k/d := by
    have a := Nat.mod_add_div r.block d
    have b := Nat.mod_add_div k d
    rw [same] at b
    apply Classical.byContradiction
    intro notLt
    have ge : k/d ≤ r.block/d := by omega
    have := Nat.mul_le_mul_left d ge
    omega
  let w : Fin 8 := ⟨4*r.group.val, by have := r.group.isLt; omega⟩
  have gw : group w = r.group := by apply Fin.ext; simp [w, group]
  intro t ht
  obtain ⟨u, hu, ut⟩ := f.acquired k t ht
  rw [CTA.acquire, f.emptyOperand _ kb] at hu
  obtain ⟨v, hv, vu⟩ := (handshake (f.slotProtocol hd (Nat.mod_lt _ hd)) u).2 _ kb' hu w _ earlier
  have given : c.give d w r.block = some v := by
    simpa [CTA.give, f.released _ _ h.bound, same] using hv
  obtain ⟨a, ha, av⟩ := f.retiredBeforeGive w h.bound _ given
  exact ⟨a, gw ▸ ha, by omega⟩

theorem read_current_generation (f : Fold c d count) (hd : 0 < d) (h : ReadCode c d count r)
    (issued : By r.issue cut) (active : r.issue = some cut ∨ ¬ By (c.done_ r.group r.block) cut) :
    (writes c d count).Shows r.slot r.block r.block cut := by
  have bound := h.bound
  obtain ⟨t, polled, le⟩ := ((f.acquiredFull r.group r.block).trans h.fenced).byTime issued
  apply Writes.shows_of_bounds _ bound ((f.destination _ bound).trans h.operand.symm)
    (f.source _ bound) ((f.startArrival _).trans (f.arrival _))
  · intro k older _
    rcases Nat.eq_or_lt_of_le older with rfl | earlier
    · exact (poll_published f hd _ bound polled).mono le
    · obtain ⟨u, hu, _⟩ := f.pollOrder r.group earlier bound t polled
      exact (poll_published f hd _ (by omega) hu).mono (by omega)
  · intro k later kb same
    exact before_overwrite (Or.inr (h.committed.trans (f.committed _ _)))
      (overwrite_after_last_read f hd h kb later same) active

end Ring.WarpSpecialized
