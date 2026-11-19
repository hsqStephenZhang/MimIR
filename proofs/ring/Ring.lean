import Ring.Fold
import Ring.Witness
import Ring.WarpWitness

namespace Ring

private theorem seq_first (f : Nat → List Op) (n : Nat) :
    ∃ tail, seq f (n + 1) = f 0 ++ tail := by
  induction n with
  | zero => exact ⟨[], by simp [seq]⟩
  | succ n ih =>
    obtain ⟨tail, h⟩ := ih
    exact ⟨tail ++ f (n + 1), by rw [seq, h, List.append_assoc]⟩

private theorem singleSlot_first (n : Nat) :
    ∃ tail, fold 1 (n + 1) lag width = .poll 0 0 :: tail := by
  obtain ⟨tail, h⟩ := seq_first (iteration 1 (n + 1) lag width) n
  unfold fold
  rw [h]
  simp [prologue, seq, iteration]

private theorem initial_still (dev : HW initial m) : False := by
  cases dev with
  | slot step => cases step <;> simp [initial, empty] at *
  | retire split => simp [initial] at split

private theorem poll_zero_stuck (step : Step d bytes (.poll 0 0 :: ops, initial) y) :
    y = (.poll 0 0 :: ops, initial) := by
  cases step with
  | hw dev => exact (initial_still dev).elim
  | retry => rfl
  | host instr => cases instr with
    | poll ok => simp [passes, initial, empty] at ok

private theorem stuck_run (run : Run d bytes x y) (stuck : x = (.poll 0 0 :: ops, initial)) :
    y = x := by
  induction run with
  | refl => rfl
  | snoc _ step ih => rw [ih, stuck] at step; exact (poll_zero_stuck step).trans stuck.symm

theorem singleSlot_stalls (run : Run 1 bytes (fold 1 (n + 1) lag width, initial) y) :
    y = (fold 1 (n + 1) lag width, initial) := by
  obtain ⟨ops, h⟩ := singleSlot_first (lag := lag) (width := width) n
  exact stuck_run run (congrArg (·, initial) h)

private theorem empty_one : Code 1 zero (fold 1 0 lag width) (finalMark 1 0) :=
  .cons .waitGroups .nil

-- Covers both certified programs and the literal one-slot program stuck at its first poll.
def FoldInv (d count : Nat) (cfg : Config) : Prop :=
  Certified d (finalMark d count) cfg ∨
    (d = 1 ∧ ∃ tail, cfg = (.poll 0 0 :: tail, initial))

theorem foldInvariant (hd : 0 < d) (run : Run d bytes (fold d count lag width, initial) cfg) :
    FoldInv d count cfg := by
  by_cases deep : 2 ≤ d
  · exact Or.inl (Certified.reachable ⟨zero, Inv.initial, foldCertificate deep⟩ run)
  have eq : d = 1 := by omega
  subst d
  cases count with
  | zero => exact Or.inl (Certified.reachable ⟨zero, Inv.initial, empty_one⟩ run)
  | succ n =>
    obtain ⟨tail, first⟩ := singleSlot_first (lag := lag) (width := width) n
    exact Or.inr ⟨rfl, tail, (singleSlot_stalls run).trans (congrArg (·, initial) first)⟩

theorem FoldInv.head (inv : FoldInv d count (op :: ops, m)) : SafeAt d m op := by
  rcases inv with cert | ⟨rfl, tail, eq⟩
  · exact cert.head
  · cases eq
    exact Or.inl rfl

/-- T1 holds before every MMA issue, including every issue in a multi-MMA group. -/
theorem read_current_generation (hd : 0 < d)
    (run : Run d bytes (fold d count lag width, initial) (.mma i s :: tail, m)) :
    (m.slots s).visible = some (ref d i) ∧
    (m.slots s).phase = i / d + 1 ∧ (m.slots s).tx = none := by
  have ready : Ready d i (m.slots s) := (foldInvariant hd run).head
  exact ⟨ready.2.1, ready.1, ready.2.2⟩

theorem live_read_race_free (hd : 0 < d)
    (run : Run d bytes (fold d count lag width, initial) cfg) (live : Live cfg.2 r) :
    (cfg.2.slots r.slot).visible = some r ∧ (cfg.2.slots r.slot).tx = none := by
  have safe : LiveReady d cfg.2 := by
    by_cases deep : 2 ≤ d
    · exact Certified.liveReady ⟨zero, Inv.initial, foldCertificate deep⟩ LiveReady.initial run
    have : d = 1 := by omega
    subst d
    cases count with
    | zero => exact Certified.liveReady ⟨zero, Inv.initial, empty_one⟩ LiveReady.initial run
    | succ n => rw [singleSlot_stalls run]; exact LiveReady.initial
  obtain ⟨tag, ready⟩ := safe r live
  exact ⟨ready.2.1.trans (congrArg some tag.symm), ready.2.2⟩

/-- T2 holds before issue, including reads pending commit and reads in any live group. -/
theorem overwrite_after_last_read (hd : 0 < d)
    (run : Run d bytes (fold d count lag width, initial) (.fill j s :: tail, m))
    (previousExists : d ≤ j) : ¬ Live m (ref d (j - d)) :=
  (foldInvariant hd run).head previousExists

theorem release_after_last_read (hd : 0 < d)
    (run : Run d bytes (fold d count lag width, initial) (.release (some j) :: tail, m)) :
    ¬ Live m (ref d j) := (foldInvariant hd run).head

/-- The parity observation cannot have skipped two flips; success identifies the exact fill. -/
theorem no_ABA (hd : 0 < d)
    (run : Run d bytes (fold d count lag width, initial) (.poll i s :: tail, m)) :
    ((m.slots s).phase = i / d ∨ (m.slots s).phase = i / d + 1) ∧
    (passes d i s m ↔ Ready d i (m.slots s)) := by
  have head := (foldInvariant hd run).head
  rcases head with phase | ready
  · refine ⟨Or.inl phase, ?_, ?_⟩
    · intro ok; exact (ok (congrArg (· % 2) phase)).elim
    · intro h; have := h.1; omega
  · refine ⟨Or.inr ready.1, fun _ => ready, ?_⟩
    intro _
    unfold passes
    rw [ready.1]
    omega

private theorem runCertificate (code : Code d zero ops (finalMark d count)) :
    ∃ cfg, Run d bytes (ops, initial) cfg ∧ Drained count cfg := by
  obtain ⟨m, run, inv, q⟩ := code.eager (bytes := bytes) Inv.initial ⟨fun _ => rfl, rfl⟩
  exact ⟨([], m), run, rfl, inv.commits, inv.pending, q⟩

theorem exists_drained (hd : 2 ≤ d) :
    ∃ cfg, Run d bytes (fold d count lag width, initial) cfg ∧ Drained count cfg :=
  runCertificate (foldCertificate hd)

/-- Exact existence criterion for the source prologue and refill order. -/
theorem exists_drained_iff (hd : 0 < d) :
    (∃ cfg, Run d bytes (fold d count lag width, initial) cfg ∧ Drained count cfg) ↔
      2 ≤ d ∨ count = 0 := by
  constructor
  · rintro ⟨cfg, run, drained⟩
    by_cases deep : 2 ≤ d
    · exact Or.inl deep
    have : d = 1 := by omega
    subst d
    cases count with
    | zero => exact Or.inr rfl
    | succ n =>
      have stuck := singleSlot_stalls run
      have done := drained.2.1
      simp [stuck, initial] at done
  · rintro (deep | rfl)
    · exact exists_drained deep
    · by_cases deep : 2 ≤ d
      · exact exists_drained deep
      have : d = 1 := by omega
      subst d
      exact runCertificate empty_one

end Ring
