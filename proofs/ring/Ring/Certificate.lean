import Ring.Invariant

namespace Ring

-- A certificate is checked against the *unannotated* host instruction list.
-- Rule premises constrain certification; Host and Step do not consult them.
inductive Code (d : Nat) : Mark → List Op → Mark → Prop
  | nil : Code d v [] v
  | cons : Rule d v op w → Code d w ops z → Code d v (op :: ops) z

theorem Code.append (a : Code d u xs v) (b : Code d v ys w) : Code d u (xs ++ ys) w := by
  induction a with
  | nil => exact b
  | cons r _ ih => exact .cons r (ih b)

def Certified (d : Nat) (endMark : Mark) (cfg : Config) :=
  ∃ v, Inv d v cfg.2 ∧ Code d v cfg.1 endMark

theorem Certified.step (h : Certified d z x) (s : Step d bytes x y) : Certified d z y := by
  obtain ⟨v, inv, code⟩ := h
  cases s with
  | hw device => exact ⟨v, inv.hardware device, code⟩
  | retry => exact ⟨v, inv, code⟩
  | host step =>
    cases code with
    | cons rule tail => exact ⟨_, inv.preserve rule step, tail⟩

theorem Certified.reachable (h : Certified d z x) (run : Run d bytes x y) : Certified d z y := by
  induction run with
  | refl => exact h
  | snoc _ step ih => exact ih.step step

def SafeAt (d : Nat) (m : Machine) : Op → Prop
  | .mma i => Ready d i (m.slots (i % d))
  | .fill j => d ≤ j → ¬ Live m (ref d (j - d))
  | .poll i => (m.slots (i % d)).phase = i / d ∨ Ready d i (m.slots (i % d))
  | .release (some j) => ¬ Live m (ref d j)
  | _ => True

private theorem Inv.retired (inv : Inv d v m) (floor : j < v.floor) (commit : j < v.k) :
    ¬ Live m (ref d j) := by
  rintro (pending | ⟨g, hg, hr⟩)
  · rw [inv.pending] at pending
    have := congrArg Ref.block (List.eq_of_mem_replicate pending)
    change j = v.k at this
    omega
  · obtain ⟨bound, _, reads⟩ := inv.groups _ hg
    have := congrArg Ref.block (reads _ hr)
    change j = g.serial at this
    omega

theorem Certified.head (h : Certified d z (op :: ops, m)) : SafeAt d m op := by
  obtain ⟨v, inv, code⟩ := h
  cases code with
  | cons rule _ =>
    have ord := inv.ordered
    cases rule with
    | fill _ old =>
      intro later
      exact inv.retired (old later).1 (old later).2
    | release retired =>
      cases hp : (‹Option Nat›) with
      | none => trivial
      | some j => exact inv.retired (retired j hp).1 (retired j hp).2
    | poll issued =>
      rcases inv.pollWindow issued with ready | ⟨phase, _⟩
      · exact Or.inr ready
      · exact Or.inl phase
    | mma lo seen =>
      simpa [SafeAt, At, seen] using inv.window v.k lo (by omega)
    | _ => trivial

-- Progress uses an explicit eager schedule. Safety above makes no eagerness assumption.
inductive Devices : Machine → Machine → Prop
  | refl : Devices m m
  | snoc : Devices m n → HW n out → Devices m out

theorem Devices.run (h : Devices m n) : Run d bytes (ops, m) (ops, n) := by
  induction h with
  | refl => exact .refl
  | snoc _ step ih => exact .snoc ih (.hw step)

theorem Devices.preserve (h : Devices m n) (inv : Inv d v m) : Inv d v n := by
  induction h with
  | refl => exact inv
  | snoc _ step ih => exact ih.hardware step

private theorem finishFill (m : Machine) (s : Nat) (p : Copy) :
    ∃ m', Devices (m.put s ((m.slots s).start p)) m' ∧
      (Quiet m → Quiet m') := by
  let started := m.put s ((m.slots s).start p)
  let announced := started.put s { started.slots s with tx := some ⟨p, true, 0⟩ }
  have a : Device (started.slots s) (announced.slots s) := by
    simp only [announced, Machine.put, ↓reduceIte]
    exact .arrive (t := ⟨p, false, 0⟩) (by simp [started, Machine.put, Cell.start]) rfl
  have update (x : Machine) (s : Nat) (c : Cell) :
      x.put s ((x.put s c).slots s) = x.put s c := by simp [Machine.put]
  have first : Devices started announced :=
    .snoc .refl (by simpa [announced, update] using HW.slot a)
  let paid := announced.put s { announced.slots s with tx := some ⟨p, true, p.bytes⟩ }
  have second : Devices started paid := by
    by_cases free : p.bytes = 0
    · have same : paid = announced := by
        simp [paid, announced, started, Machine.put, Cell.start, free, funext_iff]
      rwa [same]
    · have dev : Device (announced.slots s) (paid.slots s) := by
        simp only [paid, Machine.put, ↓reduceIte]
        exact .deliver (t := ⟨p, true, 0⟩) (by simp [announced, Machine.put])
          (Nat.pos_of_ne_zero free) (Nat.le_refl _)
      exact .snoc first (by simpa [paid, update] using HW.slot dev)
  let result := paid.put s ⟨(paid.slots s).phase + 1, some p.src, none⟩
  have pub : Device (paid.slots s) (result.slots s) := by
    simp only [result, Machine.put, ↓reduceIte]
    exact .publish (t := ⟨p, true, p.bytes⟩) (by simp [paid, Machine.put]) rfl rfl
  refine ⟨result, .snoc second (by simpa [result, update] using HW.slot pub), ?_⟩
  rintro ⟨quiet, groups⟩
  refine ⟨?_, groups⟩
  intro t
  by_cases eq : t = s <;> simp [result, paid, announced, started, Machine.put, eq, quiet]

private theorem Rule.eager (rule : Rule d v op w) (inv : Inv d v m) (q : Quiet m)
    (tail : List Op) : ∃ m', Run d bytes (op :: tail, m) (tail, m') ∧ Inv d w m' ∧ Quiet m' := by
  have single {m' : Machine} (host : Host d bytes op m m') :
      Run d bytes (op :: tail, m) (tail, m') := .snoc .refl (.host host)
  have direct {m' : Machine} (host : Host d bytes op m m') (quiet : Quiet m') :
      ∃ out, Run d bytes (op :: tail, m) (tail, out) ∧ Inv d w out ∧ Quiet out :=
    ⟨m', single host, inv.preserve rule host, quiet⟩
  cases rule with
  | fill space old =>
    obtain ⟨last, hw, quiet⟩ := finishFill m (v.filled % d) ⟨ref d v.filled, bytes v.filled⟩
    exact ⟨last, (single .fill).trans hw.run,
      hw.preserve (inv.preserve (.fill space old) .fill), quiet q⟩
  | poll issued =>
    rcases inv.pollWindow issued with ready | ⟨_, t, tx, _⟩
    · apply direct (.poll ?_) q
      unfold passes
      rw [ready.1]
      omega
    · simp [q.1 _] at tx
  | mma => exact direct .mma q
  | commit bound =>
    let next : Machine := { m with pending := [], groups := [], commits := m.commits + 1 }
    have dev : HW { next with groups := ⟨m.commits, m.pending⟩ :: m.groups } next := by
      apply HW.retire (g := ⟨m.commits, m.pending⟩) (xs := []) (ys := [])
      simp [q.2]
    have posted := inv.preserve (bytes := bytes) (.commit bound) .commit
    exact ⟨next, .snoc (single .commit) (.hw dev), posted.hardware dev, q.1, rfl⟩
  | waitGroups => exact direct (.waitGroups (by simp [q.2])) q
  | release => exact direct .release q
  | next => exact direct .next q

theorem Code.eager (code : Code d v ops z) (inv : Inv d v m) (quiet : Quiet m) :
    ∃ m', Run d bytes (ops, m) ([], m') ∧ Inv d z m' ∧ Quiet m' := by
  induction code generalizing m with
  | nil => exact ⟨m, .refl, inv, quiet⟩
  | cons rule _ ih =>
    obtain ⟨mid, head, pres, q⟩ := rule.eager (bytes := bytes) inv quiet _
    obtain ⟨last, tail, done, q'⟩ := ih pres q
    exact ⟨last, head.trans tail, done, q'⟩

end Ring
