import Ring.Async
import Ring.Cooperative

namespace Ring
open Timeline

/-- Each lane reads every real block and drains its copies. The witness uses
    synchronous loads or WGMMA with proxy publication according to the path. -/
theorem Async.exists_execution (hd : 2 ≤ d) (threads count : Nat) (proxy : Bool) :
    ∃ c : Async.CTA threads, Async.Fold c d count (d-2) proxy ∧
      (∀ lane i, i < count → ∃ r, Async.ReadCode c d count proxy r ∧ r.lane = lane ∧ r.block = i ∧
        r.reader = (if proxy then .wgmma i else .load) ∧ r.issue.isSome) ∧
      ∀ l, (c.writer l).settle.isSome := by
  let copy := fun (j off : Nat) => event (d-1+count) j (12*j + off)
  let loop := fun (i off : Nat) => event count i (12*(i+(d-1)) + off)
  let c : Async.CTA threads := {
    writer := fun _ => {
      slot := (· % d), sourceBlock := Async.source d count,
      issue := (copy · 4), commit := (copy · 5), complete := (copy · 6),
      waitGroup := (loop · 0), proxyFence := (loop · 1), settle := some (12*(count+d)) }
    fence := (loop · 2), warpgroup := fun l => l.val / 128,
    batchFence := fun _ i => loop i 7, batchCommit := fun _ i => loop i 9,
    batchDone := fun _ i => loop i 10, batchWait := fun _ i => loop i 11 }
  refine ⟨c, ?_, ?_, fun _ => rfl⟩
  · constructor <;> intros <;> simp_all [c, copy, loop, event_edge] <;> try omega
    case arrive => cases proxy <;> simp <;> omega
    case waitRule l i hi =>
      intro j t ht hc
      simp [event, hi] at ht
      subst t
      simp [By, event] at hc ⊢
      obtain ⟨_, ⟨jb, rfl⟩, time⟩ := hc
      exact ⟨_, ⟨by omega, rfl⟩, by omega⟩
    case drain => simp_all [event, Edge]; omega
  · intro lane i hi
    let r : Async.Read threads := ⟨lane, i, i%d, loop i 8, if proxy then .wgmma i else .load⟩
    refine ⟨r, ?_, rfl, rfl, rfl, by simp [r, loop, event, hi]⟩
    cases proxy <;> constructor <;> simp [c, r, copy, loop, hi, Async.afterStep] <;> try omega
    all_goals split <;> simp_all [event, Edge] <;> omega

/-- A fully executed cooperative fold, with each warpgroup reading every
    block. Small instruction times inside one iteration may be shared across
    warpgroups; completion and CTA release are distinct events. -/
theorem Cooperative.exists_execution (hd : 2 ≤ d) (depthBound : d-1 ≤ count) (leader : Fin groups) :
    ∃ c : Cooperative.CTA groups, Cooperative.Fold c d count ∧
      ∀ g i, i < count → ∃ r, Cooperative.ReadCode c d count r ∧
        r.group = g ∧ r.block = i ∧ r.issue.isSome := by
  let loop := fun (i off : Nat) => event count i (16*(i+(d-1)) + off)
  let copy := fun (j off : Nat) => event count j (16*j + 12 + off)
  let c : Cooperative.CTA groups := {
    leader, start := (copy · 0), arrived := (copy · 1), bytesComplete := (copy · 1),
    publish := (copy · 2), fullBarrier := (· % d), fillSlot := (· % d), source := id,
    pollSlot := fun _ i => i%d, poll := fun _ i => loop i 0,
    batchFence := fun _ i => loop i 1, commit := fun _ i => loop i 3,
    done_ := fun _ i => loop i 4, waitGroup := fun _ i => loop i 5,
    fence := (loop · 6), finalWait := fun _ => some (16*(count+d)) }
  refine ⟨c, ?_, ?_⟩
  · constructor <;> intros <;> try first | assumption | rfl | (simp_all [c, loop, copy] <;> omega)
    case parity g i t hi h =>
      simp [c, loop, event, hi] at h
      subst t
      -- Ahead-of-time prologue copies may already have published, but none of
      -- the next visits to i's own physical slot can have done so.
      have phase : Cooperative.phase c count (i%d) (16*(i+(d-1))) = i/d + 1 := by
        unfold Cooperative.phase phaseAt
        rw [tally_prefix (q := fun j => decide (j % d = i % d)) (i := i+1),
            Nat.min_eq_right (by omega : i+1 ≤ count), tally]
        · rw [visits (by omega) (Nat.mod_lt _ (by omega))]
          simp
        · intro k hk
          simp only [c, copy, event, hk, ite_true, Bool.and_eq_true, decide_eq_true_eq,
            occurred_iff, By, Option.some.injEq, exists_eq_left']
          constructor
          · rintro ⟨same, le⟩
            by_cases h : k ≤ i
            · exact ⟨by omega, same⟩
            · have := circle_gap (by omega : 0 < d) same.symm (by omega)
              omega
          · intro ⟨le, same⟩; exact ⟨same, by omega⟩
      change Cooperative.phase c count (i%d) _ % 2 ≠ _
      rw [phase]
      omega
    case drain g i hi =>
      simp [c, loop, event, hi, Edge]; omega
  · intro g i hi
    let r : Cooperative.Read groups := ⟨g, i, i%d, loop i 2⟩
    refine ⟨r, ?_, rfl, rfl, by simp [r, loop, event, hi]⟩
    constructor <;> simp [r, c, loop, hi]

end Ring
