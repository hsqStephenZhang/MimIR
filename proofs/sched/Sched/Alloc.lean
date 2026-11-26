import Sched.Aff

/-!
# `Alloc`: storage sized by static bounds

A `Pack` or `Stage` that uses the storage of an `Alloc` checks statically that the elements it may fill fit.
These lemmas say that what it fills at run time stays within that static count.
-/

namespace Sched

/-- The box a `Stage` fills has at most the elements `aff_box` bounds it by, the product of `aff_reach (row, cap) + 1`. -/
theorem stage_fits (vals lo ext cap : Nat → Nat) (hcap : ∀ d, ext d ≤ cap d) :
    ∀ rows : List AffE, (∀ r ∈ rows, r.WF) →
      (rows.map fun r => (affBox vals lo ext r).2).foldr (· * ·) 1 ≤ (rows.map fun r => r.reach cap + 1).foldr (· * ·) 1
  | [], _ => Nat.le_refl _
  | r :: rows, h => by
    simp only [List.map_cons, List.foldr_cons]
    refine Nat.mul_le_mul ?_ (stage_fits vals lo ext cap hcap rows fun r' hr' => h r' (by simp [hr']))
    have := AffE.span_le_reach vals lo ext cap hcap r (h r (by simp))
    simp only [affBox]
    omega

/-- A tile of an axis bounded by `cap` reads panels below the count `levels_pack_one` allocates. -/
theorem pack_fits {x cap w : Nat} (hx : x < cap) :
    x / w < cap / w + (if cap % w ≠ 0 then 1 else 0) := by
  by_cases h : cap % w = 0
  · have hc : cap = w * (cap / w) := by
      have := Nat.div_add_mod cap w
      omega
    have : x / w < cap / w := Nat.div_lt_of_lt_mul (by omega)
    simp [h, this]
  · have := Nat.div_le_div_right (c := w) (Nat.le_of_lt hx)
    simp only [ne_eq, h, not_false_eq_true, ite_true]
    omega

end Sched
