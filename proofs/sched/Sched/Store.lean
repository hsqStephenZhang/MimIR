import Sched.Need
import Sched.View

/-!
# The storage of an `Alloc` and the reads of an edge tile

A buffer is a value in the rest of the model.
This file adds the two facts about memory that the value semantics cannot state.

The storage of an `Alloc` is written by the outermost `Pack` or `Stage` of its slot and read by the leaves below that node.
The trace of a run shows that every such read sees the last write: the tiles of a `Split` refill the storage in turn, and no node below a writer writes it again.

An edge tile runs its kernel over the static `cap` of its output loops and so reads beyond its extent.
`echeck` is a sufficient condition for these reads to stay inside a panel, which `tensor.mim` leaves to the policy.
-/

namespace Sched

/-- A write of the storage by the node that ran on the box, or a leaf's read that expects the write of that box. -/
inductive Ev where
  | write (id : Box)
  | read (id : Box)

/-- The events of one slot's storage: `held` tells that an `Alloc` above offers it, `cur` which write the slot's view reads, and `fills` whether the slot is still a producer. -/
def trace (slot : Nat) : Tree → (held : Bool) → (cur : Option Box) → (fills : Bool) → Box → List Ev
  | .leaf _, _, cur, _, _ => match cur with | some id => [.read id] | none => []
  | .split d ts _ body, held, cur, fills, b =>
    match b[d]? with
    | some (lo, n) => (tiles ts lo n).flatMap fun tl => trace slot body held cur fills (b.set d tl)
    | none => []
  | .reduce _ _ body, held, cur, fills, b => trace slot body held cur fills b
  | .pack _ _ s _ body, held, cur, fills, b =>
    if s = slot then
      if held then .write b :: trace slot body false (some b) false b else trace slot body false none false b
    else trace slot body held cur fills b
  | .stage s body, held, cur, fills, b =>
    if s = slot ∧ fills = true then
      if held then .write b :: trace slot body false (some b) false b else trace slot body false none false b
    else trace slot body held cur fills b
  | .alloc s body, held, cur, fills, b => trace slot body (held || s = slot) cur fills b
  | .switch d c small big, held, cur, fills, b =>
    match b[d]? with
    | some (_, n) => if n < c then trace slot small held cur fills b else trace slot big held cur fills b
    | none => []

/-- Replays the events on a storage that holds the write `last`: `none` if a read expects another write. -/
def replay : Option Box → List Ev → Option (Option Box)
  | last, [] => some last
  | _, .write id :: evs => replay (some id) evs
  | last, .read id :: evs => if last = some id then replay last evs else none

theorem replay_append (last : Option Box) (l₁ l₂ : List Ev) :
    replay last (l₁ ++ l₂) = (replay last l₁).bind fun last' => replay last' l₂ := by
  induction l₁ generalizing last with
  | nil => rfl
  | cons e l ih =>
    cases e with
    | write id => exact ih _
    | read id =>
      simp only [List.cons_append, replay]
      split
      · exact ih _
      · rfl

/-- Parts that each replay and keep a property of the storage do so in a row. -/
theorem replay_flatMap {τ : Type} (f : τ → List Ev) (P : Option Box → Prop) :
    ∀ (l : List τ) (last : Option Box), P last →
      (∀ x ∈ l, ∀ last, P last → ∃ last', replay last (f x) = some last' ∧ P last') →
      ∃ last', replay last (l.flatMap f) = some last' ∧ P last'
  | [], last, hp, _ => ⟨last, rfl, hp⟩
  | x :: l, last, hp, h => by
    obtain ⟨l1, h1, p1⟩ := h x (by simp) last hp
    obtain ⟨l2, h2, p2⟩ := replay_flatMap f P l l1 p1 fun y hy => h y (by simp [hy])
    refine ⟨l2, ?_, p2⟩
    rw [List.flatMap_cons, replay_append, h1]
    exact h2

/-- Below one `Alloc`, every read of its storage sees the write it expects, and a subtree that is offered no storage leaves it as it is.
The tiles of a `Split` write the storage in turn; below a writer the storage is no longer on offer, so the writer's leaves all read its write. -/
theorem trace_safe (slot : Nat) : ∀ (t : Tree) (held : Bool) (cur : Option Box) (fills : Bool) (b : Box) (last : Option Box),
    t.allocFree slot = true → (∀ id, cur = some id → held = false ∧ last = some id) →
    ∃ last', replay last (trace slot t held cur fills b) = some last' ∧ (held = false → last' = last)
  | .leaf _, held, cur, fills, b, last, _, hc => by
    cases cur with
    | none => exact ⟨last, rfl, fun _ => rfl⟩
    | some id => exact ⟨last, by simp [trace, replay, (hc id rfl).2], fun _ => rfl⟩
  | .reduce _ _ body, held, cur, fills, b, last, hf, hc => trace_safe slot body held cur fills b last hf hc
  | .alloc s body, held, cur, fills, b, last, hf, hc => by
    simp only [Tree.allocFree, Bool.and_eq_true, decide_eq_true_eq] at hf
    have : (held || decide (s = slot)) = held := by simp [hf.1]
    simp only [trace, this]
    exact trace_safe slot body held cur fills b last hf.2 hc
  | .switch d c small big, held, cur, fills, b, last, hf, hc => by
    simp only [Tree.allocFree, Bool.and_eq_true] at hf
    simp only [trace]
    split
    · split
      · exact trace_safe slot small held cur fills b last hf.1 hc
      · exact trace_safe slot big held cur fills b last hf.2 hc
    · exact ⟨last, rfl, fun _ => rfl⟩
  | .split d ts mode body, held, cur, fills, b, last, hf, hc => by
    simp only [trace]
    split
    · by_cases hh : held = false
      · obtain ⟨l', h', p'⟩ := replay_flatMap (fun tl => trace slot body held cur fills (b.set d tl)) (· = last) _ last rfl
          fun tl _ last' hl => by
            obtain ⟨l'', h'', k''⟩ := trace_safe slot body held cur fills (b.set d tl) last' hf (hl ▸ hc)
            exact ⟨l'', h'', (k'' hh).trans hl⟩
        exact ⟨l', h', fun _ => p'⟩
      · have hcn : ∀ last' id, cur = some id → held = false ∧ last' = some id := fun _ id hid =>
          absurd (hc id hid).1 hh
        obtain ⟨l', h', _⟩ := replay_flatMap (fun tl => trace slot body held cur fills (b.set d tl)) (fun _ => True) _
          last trivial fun tl _ last' _ => by
            obtain ⟨l'', h'', _⟩ := trace_safe slot body held cur fills (b.set d tl) last' hf (hcn last')
            exact ⟨l'', h'', trivial⟩
        exact ⟨l', h', fun h => absurd h hh⟩
    · exact ⟨last, rfl, fun _ => rfl⟩
  | .pack _ _ s _ body, held, cur, fills, b, last, hf, hc => by
    simp only [trace]
    split
    · split
      · rename_i hh
        obtain ⟨l', h', _⟩ := trace_safe slot body false (some b) false b (some b) hf fun id hid =>
          ⟨rfl, congrArg some (Option.some.inj hid)⟩
        exact ⟨l', by simpa [replay] using h', fun h => by rw [h] at hh; cases hh⟩
      · obtain ⟨l', h', k'⟩ := trace_safe slot body false none false b last hf (fun _ h => by cases h)
        exact ⟨l', h', fun _ => k' rfl⟩
    · exact trace_safe slot body held cur fills b last hf hc
  | .stage s body, held, cur, fills, b, last, hf, hc => by
    simp only [trace]
    split
    · split
      · rename_i hh
        obtain ⟨l', h', _⟩ := trace_safe slot body false (some b) false b (some b) hf fun id hid =>
          ⟨rfl, congrArg some (Option.some.inj hid)⟩
        exact ⟨l', by simpa [replay] using h', fun h => by rw [h] at hh; cases hh⟩
      · obtain ⟨l', h', k'⟩ := trace_safe slot body false none false b last hf (fun _ h => by cases h)
        exact ⟨l', h', fun _ => k' rfl⟩
    · exact trace_safe slot body held cur fills b last hf hc

/-- An `Alloc` whose slot has no second `Alloc` below: its storage is read consistently, whatever it held before. -/
theorem alloc_safe (slot : Nat) (body : Tree) (fills : Bool) (b : Box) (last : Option Box)
    (hf : body.allocFree slot = true) : (replay last (trace slot (.alloc slot body) false none fills b)).isSome = true := by
  obtain ⟨l', h', _⟩ := trace_safe slot body true none fills b last hf (fun _ h => by cases h)
  simp [trace, h']

/-- A buffer of the slot is named by the box of the `Pack` or `Stage` that allocates it and by how many of them lie above it. -/
abbrev Buf := Box × Nat

/-- A `Pack` or `Stage` allocates its buffer and frees it after its body; a leaf uses the buffer its view reads. -/
inductive LEv where
  | alloc (id : Buf)
  | free (id : Buf)
  | use (id : Buf)

/-- The lifetime events of one slot's buffers: `cur` is the buffer the slot's view reads, and `depth` counts the allocations above. -/
def life (slot : Nat) : Tree → (cur : Option Buf) → (fills : Bool) → (depth : Nat) → Box → List LEv
  | .leaf _, cur, _, _, _ => match cur with | some id => [.use id] | none => []
  | .split d ts _ body, cur, fills, depth, b =>
    match b[d]? with
    | some (lo, n) => (tiles ts lo n).flatMap fun tl => life slot body cur fills depth (b.set d tl)
    | none => []
  | .reduce _ _ body, cur, fills, depth, b => life slot body cur fills depth b
  | .pack _ _ s _ body, cur, fills, depth, b =>
    if s = slot then .alloc (b, depth) :: life slot body (some (b, depth)) false (depth + 1) b ++ [.free (b, depth)]
    else life slot body cur fills depth b
  | .stage s body, cur, fills, depth, b =>
    if s = slot ∧ fills = true then
      .alloc (b, depth) :: life slot body (some (b, depth)) false (depth + 1) b ++ [.free (b, depth)]
    else life slot body cur fills depth b
  | .alloc _ body, cur, fills, depth, b => life slot body cur fills depth b
  | .switch d c small big, cur, fills, depth, b =>
    match b[d]? with
    | some (_, n) => if n < c then life slot small cur fills depth b else life slot big cur fills depth b
    | none => []

/-- Replays the events on the stack of live buffers: `none` if a buffer is used while it is not live, or freed out of order. -/
def live : List Buf → List LEv → Option (List Buf)
  | st, [] => some st
  | st, .alloc id :: evs => live (id :: st) evs
  | [], .free _ :: _ => none
  | top :: st, .free id :: evs => if top = id then live st evs else none
  | st, .use id :: evs => if id ∈ st then live st evs else none

theorem live_append (st : List Buf) (l₁ l₂ : List LEv) :
    live st (l₁ ++ l₂) = (live st l₁).bind fun st' => live st' l₂ := by
  induction l₁ generalizing st with
  | nil => rfl
  | cons e l ih =>
    cases e with
    | alloc id => exact ih _
    | free id =>
      cases st with
      | nil => rfl
      | cons top st =>
        simp only [List.cons_append, live]
        split
        · exact ih _
        · rfl
    | use id =>
      simp only [List.cons_append, live]
      split
      · exact ih _
      · rfl

theorem live_flatMap {τ : Type} (f : τ → List LEv) (st : List Buf) :
    ∀ l : List τ, (∀ x ∈ l, live st (f x) = some st) → live st (l.flatMap f) = some st
  | [], _ => rfl
  | x :: l, h => by
    rw [List.flatMap_cons, live_append, h x (by simp)]
    exact live_flatMap f st l fun y hy => h y (by simp [hy])

/-- No leaf uses a buffer that is not live, and every `Pack` and `Stage` frees its buffer after its last use, in the order of allocation. -/
theorem life_safe (slot : Nat) : ∀ (t : Tree) (cur : Option Buf) (fills : Bool) (depth : Nat) (b : Box) (st : List Buf),
    (∀ id, cur = some id → id ∈ st) → live st (life slot t cur fills depth b) = some st
  | .leaf _, cur, _, _, _, st, hc => by
    cases cur with
    | none => rfl
    | some id => simp [life, live, hc id rfl]
  | .reduce _ _ body, cur, fills, depth, b, st, hc => life_safe slot body cur fills depth b st hc
  | .alloc _ body, cur, fills, depth, b, st, hc => life_safe slot body cur fills depth b st hc
  | .switch d c small big, cur, fills, depth, b, st, hc => by
    simp only [life]
    split
    · split
      · exact life_safe slot small cur fills depth b st hc
      · exact life_safe slot big cur fills depth b st hc
    · rfl
  | .split d ts mode body, cur, fills, depth, b, st, hc => by
    simp only [life]
    split
    · exact live_flatMap _ st _ fun tl _ => life_safe slot body cur fills depth (b.set d tl) st hc
    · rfl
  | .pack _ _ s _ body, cur, fills, depth, b, st, hc => by
    simp only [life]
    split
    · have := life_safe slot body (some (b, depth)) false (depth + 1) b ((b, depth) :: st)
        (fun id hid => by cases hid; simp)
      simp [live, live_append, this]
    · exact life_safe slot body cur fills depth b st hc
  | .stage s body, cur, fills, depth, b, st, hc => by
    simp only [life]
    split
    · have := life_safe slot body (some (b, depth)) false (depth + 1) b ((b, depth) :: st)
        (fun id hid => by cases hid; simp)
      simp [live, live_append, this]
    · exact life_safe slot body cur fills depth b st hc

/-- The checks that keep the reads of an edge tile inside a panel: a slot that reads a run-time output loop of a leaf is packed along it, in panels at least as wide as the tile's `cap`. -/
def echeck (ro : Nat) : Tree → AState → List Ghost → Bool
  | .leaf vdim, a, gs =>
    !(dynAt a vdim || dynAt a (vdim + 1)) ||
      (List.range ro).all fun e =>
        match a[e]? with
        | some x => !x.dyn || gs.all fun g => !g.reads e || x.cap ≤ g.unit e
        | none => true
  | .split d ts mode body, a, gs =>
    match a[d]? with
    | some x => (atiles ts mode x).all fun y => echeck ro body (a.set d y) gs
    | none => true
  | .reduce _ _ body, a, gs => echeck ro body a gs
  | .pack d w s _ body, a, gs =>
    match gs[s]? with
    | some g => echeck ro body a (gs.set s (packGhost d w g))
    | none => true
  | .stage s body, a, gs =>
    match gs[s]? with
    | some g => echeck ro body a (gs.set s (if g.ready then g else stageGhost g))
    | none => true
  | .alloc _ body, a, gs => echeck ro body a gs
  | .switch d c small big, a, gs =>
    match a[d]? with
    | some x =>
      if x.dyn then echeck ro small (a.set d (asmall c x)) gs && echeck ro big a gs
      else if x.cap < c then echeck ro small (a.set d (asmall c x)) gs else echeck ro big a gs
    | none => true

/-- Under `echeck`, an edge tile that starts at panel `q` of a loop of extent `n ≤ cap` reads lanes of panel `q` only, which lies below the count `levels_pack_one` allocates. -/
theorem edge_panel {q o w n cap tcap : Nat} (hw : 0 < w) (ho : o < tcap) (ht : tcap ≤ w) (hq : q * w < n + w)
    (hn : n ≤ cap) : q + o / w = q ∧ q < AffE.up w cap + 1 := by
  have h0 : o / w = 0 := Nat.div_eq_of_lt (by omega)
  refine ⟨by omega, ?_⟩
  have hc := AffE.le_mul_up hw cap
  have : q * w < (AffE.up w cap + 1) * w := by
    rw [Nat.add_mul, Nat.mul_comm (AffE.up w cap) w]
    omega
  exact Nat.lt_of_mul_lt_mul_right this

end Sched
