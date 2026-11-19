import Ring.WarpSpecialized

namespace Ring.WarpSpecialized
open Timeline EmptyBarrier

/-- The empty barrier with lag one admits a complete tile at depths ≥ 2,
    including the final delayed release of all eight warp leaders. -/
theorem exists_execution (hd : 2 ≤ d) (count : Nat) :
    ∃ c : CTA, Fold c d count ∧
      (∀ g i, i < count → ∃ r, ReadCode c d count r ∧ r.group = g ∧ r.block = i ∧ r.issue.isSome) ∧
      ∀ w i, i < count → (c.give d w i).isSome := by
  let loop := fun (j off : Nat) => event count j (12*j + off)
  let c : CTA := {
    slots := fun s => {
      acquire := fun h => loop (s+d*h) 0, start := fun h => loop (s+d*h) 1,
      publish := fun h => loop (s+d*h) 3, poll := fun _ h => loop (s+d*h) 4,
      give := fun _ h => loop (s+d*h) 22 }
    emptySlot := (· % d), panel := (· % d), fullSlot := (· % d),
    pollSlot := fun _ j => j%d, giveSlot := fun _ j => j%d, source := id,
    arrived := (loop · 2), bytesComplete := (loop · 2), batchFence := fun _ j => loop j 5,
    commit := fun _ j => loop j 7, done_ := fun _ j => loop j 8,
    waitGroup := fun _ j => loop j 9, finalWait := fun _ => some (12*count+9) }
  have gap {a b : Nat} (lt : a < b) : d*a+d ≤ d*b := by
    simpa [Nat.mul_succ] using Nat.mul_le_mul_left d lt
  refine ⟨c, ?_, ?_, ?_⟩
  · constructor <;> intros <;> try first | rfl |
      (simp_all [c, CTA.acquire, CTA.start, CTA.poll, CTA.give, CTA.publish, loop, Nat.mod_add_div] <;> omega)
    case full g j t jb atPoll slot =>
      simp [CTA.poll, c, loop, event, Nat.mod_add_div, jb] at atPoll
      subst t
      have bound : j/d < turns d count (j%d) :=
        (inTurns (by omega)).mpr (by simpa [Nat.mod_add_div] using jb)
      have phase : fullPhase (c.slots (j%d)) (turns d count (j%d)) (12*j+4) = j/d+1 := by
        apply fullPhase_eq (by omega)
        intro k kb
        have valid := (inTurns (by omega : 0 < d)).mp kb
        simp only [c, loop, event, valid, ite_true, By, Option.some.injEq, exists_eq_left']
        have eq := Nat.mod_add_div j d
        constructor
        · intro le
          by_cases h : j/d < k
          · have := gap h; omega
          · omega
        · intro lt
          have := Nat.mul_le_mul_left d (by omega : k ≤ j/d)
          omega
      change fullPhase (c.slots (j%d)) _ _ % 2 ≠ _
      rw [phase]
      omega
    case empty j t jb atAcq slot =>
      simp [CTA.acquire, c, loop, event, Nat.mod_add_div, jb] at atAcq
      subst t
      have bound : j/d < turns d count (j%d) :=
        (inTurns (by omega)).mpr (by simpa [Nat.mod_add_div] using jb)
      have phase : emptyPhase (c.slots (j%d)) (turns d count (j%d)) (12*j) = j/d := by
        apply emptyPhase_eq (by omega)
        intro w k kb
        have valid := (inTurns (by omega : 0 < d)).mp kb
        simp only [c, loop, event, valid, ite_true, By, Option.some.injEq, exists_eq_left']
        have eq := Nat.mod_add_div j d
        constructor
        · intro le
          by_cases h : j/d ≤ k
          · have := Nat.mul_le_mul_left d h; omega
          · omega
        · intro lt; have := gap lt; omega
      change emptyPhase (c.slots (j%d)) _ _ % 2 ≠ _
      rw [phase]
      omega
    case drain g j jb =>
      simp [c, loop, event, jb, Edge]; omega
    case giveLast w j last =>
      have jb : j < count := by omega
      simp [c, CTA.give, loop, event, Nat.mod_add_div, jb, Edge]; omega
  · intro g i hi
    let r : Read := ⟨g, i, i%d, loop i 6⟩
    refine ⟨r, ?_, rfl, rfl, by simp [r, loop, event, hi]⟩
    constructor <;> simp [r, c, loop, hi]
  · intro w i hi
    simp [c, CTA.give, loop, event, Nat.mod_add_div, hi]

end Ring.WarpSpecialized
