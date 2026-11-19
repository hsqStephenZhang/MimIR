import Std

namespace Ring

structure Ref where
  block : Nat
  slot : Nat
  generation : Nat
  deriving DecidableEq

def ref (d i : Nat) : Ref := ⟨i, i % d, i / d⟩

structure Copy where
  src : Ref
  bytes : Nat

structure Transfer where
  copy : Copy
  arrived : Bool
  received : Nat

structure Cell where
  phase : Nat
  visible : Option Ref
  tx : Option Transfer

def empty : Cell := ⟨0, none, none⟩

def Cell.start (c : Cell) (p : Copy) : Cell :=
  { c with visible := none, tx := some ⟨p, false, 0⟩ }

-- Phase is a ghost counter. The poll instruction below sees just its low bit.
inductive Device : Cell → Cell → Prop
  | arrive (h : c.tx = some t) (first : t.arrived = false) :
    Device c { c with tx := some { t with arrived := true } }
  | deliver (h : c.tx = some t) (more : t.received < n) (bound : n ≤ t.copy.bytes) :
    Device c { c with tx := some { t with received := n } }
  | publish (h : c.tx = some t) (arrival : t.arrived = true) (bytes : t.received = t.copy.bytes) :
    Device c ⟨c.phase + 1, some t.copy.src, none⟩

structure Group where
  serial : Nat
  reads : List Ref

structure Machine where
  slots : Nat → Cell
  pending : List Ref
  groups : List Group
  commits : Nat

def initial : Machine := ⟨fun _ => empty, [], [], 0⟩

def Machine.put (m : Machine) (s : Nat) (c : Cell) : Machine :=
  { m with slots := fun t => if t = s then c else m.slots t }

inductive Op where
  | fill (j slot : Nat)
  | poll (i slot : Nat)
  | mma (i slot : Nat)
  | commit
  | waitGroups (n : Nat)
  | fence
  | wgmmaFence
  | release (previous : Option Nat)
  | next
  deriving DecidableEq

-- HW may run at any program point. Groups complete independently of one another.
inductive HW : Machine → Machine → Prop
  | slot (step : Device (m.slots s) c) : HW m (m.put s c)
  | retire (split : m.groups = xs ++ g :: ys) : HW m { m with groups := xs ++ ys }

def passes (d i slot : Nat) (m : Machine) : Prop :=
  (m.slots slot).phase % 2 ≠ (i / d) % 2

-- No read-generation or reuse checks occur at fill or MMA issue.
inductive Host (d : Nat) (bytes : Nat → Nat) : Op → Machine → Machine → Prop
  | fill : Host d bytes (.fill j s) m
      (m.put s ((m.slots s).start ⟨⟨j, s, j / d⟩, bytes j⟩))
  | poll (ok : passes d i s m) : Host d bytes (.poll i s) m m
  | mma : Host d bytes (.mma i s) m { m with pending := ⟨i, s, i / d⟩ :: m.pending }
  | commit : Host d bytes .commit m
      { m with pending := [], groups := ⟨m.commits, m.pending⟩ :: m.groups, commits := m.commits + 1 }
  | waitGroups (recent : ∀ g ∈ m.groups, m.commits ≤ g.serial + n) :
      Host d bytes (.waitGroups n) m m
  | wgmmaFence : Host d bytes .wgmmaFence m m
  | fence : Host d bytes .fence m m
  | release : Host d bytes (.release prev) m m
  | next : Host d bytes .next m m

abbrev Config := List Op × Machine

inductive Step (d : Nat) (bytes : Nat → Nat) : Config → Config → Prop
  | hw (h : HW m m') : Step d bytes (ops, m) (ops, m')
  | retry (h : ¬ passes d i s m) : Step d bytes (.poll i s :: ops, m) (.poll i s :: ops, m)
  | host (h : Host d bytes op m m') : Step d bytes (op :: ops, m) (ops, m')

inductive Run (d : Nat) (bytes : Nat → Nat) : Config → Config → Prop
  | refl : Run d bytes x x
  | snoc : Run d bytes x y → Step d bytes y z → Run d bytes x z

theorem Run.trans (a : Run d bytes x y) (b : Run d bytes y z) : Run d bytes x z := by
  induction b with
  | refl => exact a
  | snoc _ step ih => exact .snoc ih step

def Live (m : Machine) (r : Ref) : Prop :=
  r ∈ m.pending ∨ ∃ g ∈ m.groups, r ∈ g.reads

def Quiet (m : Machine) : Prop :=
  (∀ s, (m.slots s).tx = none) ∧ m.groups = []

def Drained (count : Nat) (c : Config) : Prop :=
  c.1 = [] ∧ c.2.commits = count ∧ c.2.pending = [] ∧ Quiet c.2

end Ring
