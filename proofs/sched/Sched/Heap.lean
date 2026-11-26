import Sched.Bounds

/-!
# A heap with addresses

A block has an address, a capacity, the shape it currently holds data in, and a tag that names the fill it holds.
A run of the machine is a list of events on the heap, and a run is safe if its events replay without a fault:
no read out of bounds, no use of a freed block, no read of a block that another fill has overwritten, no fill beyond a capacity.
-/

namespace Sched

abbrev Addr := Nat

structure Blk where
  cap   : Nat
  shape : List Nat
  tag   : Option Box

abbrev Heap := Addr → Option Blk

inductive HEv where
  | alloc (id : Addr) (cap : Nat)
  | free (id : Addr)
  /-- Gives the block a shape and writes all of it; `tag` names the box of the node that fills it. -/
  | fill (id : Addr) (shape : List Nat) (tag : Box)
  /-- Reads the block at `idx`, expecting the fill `tag`. -/
  | read (id : Addr) (tag : Box) (idx : List Nat)
  /-- A read of a slot that holds no data. -/
  | fault

def prod (l : List Nat) : Nat := l.foldr (· * ·) 1

def inShape : List Nat → List Nat → Bool
  | [], [] => true
  | i :: idx, n :: shape => decide (i < n) && inShape idx shape
  | _, _ => false

theorem inShape_map {τ : Type} (f g : τ → Nat) : ∀ l : List τ, (∀ x ∈ l, f x < g x) → inShape (l.map f) (l.map g) = true
  | [], _ => rfl
  | x :: l, h => by
    simp only [List.map_cons, inShape, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨h x (by simp), inShape_map f g l fun y hy => h y (by simp [hy])⟩

theorem inShape_append : ∀ {i₁ s₁ i₂ s₂ : List Nat}, inShape i₁ s₁ = true → inShape i₂ s₂ = true →
    inShape (i₁ ++ i₂) (s₁ ++ s₂) = true
  | [], [], _, _, _, h => h
  | _ :: _, _ :: _, _, _, h₁, h₂ => by
    simp only [inShape, Bool.and_eq_true] at h₁
    simp only [List.cons_append, inShape, Bool.and_eq_true]
    exact ⟨h₁.1, inShape_append h₁.2 h₂⟩
  | [], _ :: _, _, _, h, _ => by simp [inShape] at h
  | _ :: _, [], _, _, h, _ => by simp [inShape] at h

def hset (h : Heap) (id : Addr) (v : Option Blk) : Heap := fun i => if i = id then v else h i

def hstep (h : Heap) : HEv → Option Heap
  | .alloc id cap =>
    match h id with
    | none => some (hset h id (some ⟨cap, [], none⟩))
    | some _ => none
  | .free id =>
    match h id with
    | some _ => some (hset h id none)
    | none => none
  | .fill id shape tag =>
    match h id with
    | some blk => if prod shape ≤ blk.cap then some (hset h id (some ⟨blk.cap, shape, some tag⟩)) else none
    | none => none
  | .read id tag idx =>
    match h id with
    | some blk => if blk.tag = some tag ∧ inShape idx blk.shape = true then some h else none
    | none => none
  | .fault => none

def hreplay : Heap → List HEv → Option Heap
  | h, [] => some h
  | h, e :: es => (hstep h e).bind fun h' => hreplay h' es

theorem hreplay_append (h : Heap) (l₁ l₂ : List HEv) :
    hreplay h (l₁ ++ l₂) = (hreplay h l₁).bind fun h' => hreplay h' l₂ := by
  induction l₁ generalizing h with
  | nil => rfl
  | cons e l ih =>
    simp only [List.cons_append, hreplay]
    cases hstep h e with
    | none => rfl
    | some h' => exact ih h'

/-- Parts that each replay and keep a property of the heap do so in a row. -/
theorem hreplay_flatMap {τ : Type} (f : τ → List HEv) (P : Heap → Prop) :
    ∀ (l : List τ) (h : Heap), P h → (∀ x ∈ l, ∀ h, P h → ∃ h', hreplay h (f x) = some h' ∧ P h') →
      ∃ h', hreplay h (l.flatMap f) = some h' ∧ P h'
  | [], h, hp, _ => ⟨h, rfl, hp⟩
  | x :: l, h, hp, hx => by
    obtain ⟨h1, e1, p1⟩ := hx x (by simp) h hp
    obtain ⟨h2, e2, p2⟩ := hreplay_flatMap f P l h1 p1 fun y hy => hx y (by simp [hy])
    exact ⟨h2, by rw [List.flatMap_cons, hreplay_append, e1]; exact e2, p2⟩

/-- Reads leave the heap as it is, so a list of reads replays if each of them is allowed. -/
theorem hreplay_reads {τ : Type} (h : Heap) (id : Addr) (tag : Box) (blk : Blk) (hb : h id = some blk)
    (ht : blk.tag = some tag) (idx : τ → List Nat) :
    ∀ l : List τ, (∀ x ∈ l, inShape (idx x) blk.shape = true) → hreplay h (l.map fun x => .read id tag (idx x)) = some h
  | [], _ => rfl
  | x :: l, hx => by
    simp only [List.map_cons, hreplay, hstep, hb, ht, hx x (by simp), and_self, ite_true, Option.bind_some]
    exact hreplay_reads h id tag blk hb ht idx l fun y hy => hx y (by simp [hy])

theorem hset_hset (h : Heap) (id : Addr) (v w : Option Blk) : hset (hset h id v) id w = hset h id w := by
  funext i
  simp only [hset]
  split <;> rfl

theorem hset_self (h : Heap) (id : Addr) : hset h id (h id) = h := by
  funext i
  simp only [hset]
  split
  · rename_i e; rw [e]
  · rfl

theorem hset_same (h : Heap) (id : Addr) (v : Option Blk) : hset h id v id = v := by simp [hset]

theorem hset_ne (h : Heap) {id i : Addr} (v : Option Blk) (hne : i ≠ id) : hset h id v i = h i := by simp [hset, hne]

end Sched
