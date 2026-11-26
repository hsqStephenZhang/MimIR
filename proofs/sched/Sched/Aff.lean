/-!
# Access maps as data

A model of `AffE`, `aff_eval`, `aff_rel`, `aff_span` and `aff_reach` of `tensor.mim`, and the soundness of the box a `Stage` derives from them.
Loops and symbols are plain naturals here, where `tensor.mim` has `Idx n` and `Idx s`.
-/

namespace Sched

inductive AffE where
  | dim  (d : Nat)
  | sym  (k : Nat)
  | lit  (c : Nat)
  | add  (x y : AffE)
  | mul  (x : AffE) (c : Nat)
  | fdiv (x : AffE) (c : Nat)
  | cdiv (x : AffE) (c : Nat)
  | mod  (x : AffE) (c : Nat)
  deriving Repr, DecidableEq

namespace AffE

def up (c x : Nat) : Nat := (x + (c - 1)) / c

def eval (vals x : Nat → Nat) : AffE → Nat
  | dim d    => x d
  | sym k    => vals k
  | lit c    => c
  | add a b  => eval vals x a + eval vals x b
  | mul a c  => eval vals x a * c
  | fdiv a c => eval vals x a / c
  | cdiv a c => up c (eval vals x a)
  | mod a c  => eval vals x a % c

/-- The terms in `x` without the constants, and whether there are any. -/
def rel (x : Nat → Nat) : AffE → Bool × Nat
  | dim d   => (true, x d)
  | add a b =>
    match rel x a, rel x b with
    | (false, _), (false, _) => (false, 0)
    | (false, _), (true, v)  => (true, v)
    | (true, u), (false, _)  => (true, u)
    | (true, u), (true, v)   => (true, u + v)
  | mul a c => ((rel x a).1, (rel x a).2 * c)
  | _       => (false, 0)

/-- Whether the expression is affine, the start of its interval over the box `(lo, ext)`, and the interval's width minus one. -/
def span (vals lo ext : Nat → Nat) : AffE → Bool × Nat × Nat
  | dim d    => (true, lo d, ext d - 1)
  | sym k    => (true, vals k, 0)
  | lit c    => (true, c, 0)
  | add a b  =>
    ((span vals lo ext a).1 && (span vals lo ext b).1,
      (span vals lo ext a).2.1 + (span vals lo ext b).2.1,
      (span vals lo ext a).2.2 + (span vals lo ext b).2.2)
  | mul a c  => ((span vals lo ext a).1, (span vals lo ext a).2.1 * c, (span vals lo ext a).2.2 * c)
  | fdiv a c =>
    (false, (span vals lo ext a).2.1 / c,
      ((span vals lo ext a).2.1 + (span vals lo ext a).2.2) / c - (span vals lo ext a).2.1 / c)
  | cdiv a c =>
    (false, up c (span vals lo ext a).2.1,
      up c ((span vals lo ext a).2.1 + (span vals lo ext a).2.2) - up c (span vals lo ext a).2.1)
  | mod a c  =>
    if c < (span vals lo ext a).2.1 % c + (span vals lo ext a).2.2 + 1 then (false, 0, c - 1)
    else (false, (span vals lo ext a).2.1 % c, (span vals lo ext a).2.2)

/-- A bound on the width minus one of `span` that does not depend on `lo`. -/
def reach (ext : Nat → Nat) : AffE → Nat
  | dim d    => ext d - 1
  | sym _    => 0
  | lit _    => 0
  | add a b  => reach ext a + reach ext b
  | mul a c  => reach ext a * c
  | fdiv a c => up c (reach ext a)
  | cdiv a c => up c (reach ext a)
  | mod a c  => if reach ext a = 0 then 0 else c - 1

/-- Whether the expression mentions loop `k`, also below a division or a modulus. -/
def mentions : AffE → Nat → Bool
  | dim d, k    => d == k
  | add a b, k  => mentions a k || mentions b k
  | mul a _, k  => mentions a k
  | fdiv a _, k => mentions a k
  | cdiv a _, k => mentions a k
  | mod a _, k  => mentions a k
  | _, _        => false

/-- Divisors and moduli are positive. -/
def WF : AffE → Prop
  | add a b  => WF a ∧ WF b
  | mul a _  => WF a
  | fdiv a c => WF a ∧ 0 < c
  | cdiv a c => WF a ∧ 0 < c
  | mod a c  => WF a ∧ 0 < c
  | _        => True

theorem up_mono (c : Nat) {a b : Nat} (h : a ≤ b) : up c a ≤ up c b :=
  Nat.div_le_div_right (Nat.add_le_add_right h _)

theorem le_mul_up {c : Nat} (hc : 0 < c) (a : Nat) : a ≤ c * up c a := by
  unfold up
  have h1 := Nat.div_add_mod (a + (c - 1)) c
  have h2 := Nat.mod_lt (a + (c - 1)) hc
  omega

theorem up_mul {c : Nat} (hc : 0 < c) (k : Nat) : up c (c * k) = k := by
  unfold up
  rw [Nat.mul_add_div hc, Nat.div_eq_of_lt (by omega)]
  rfl

theorem up_add_le {c : Nat} (hc : 0 < c) (a b : Nat) : up c (a + b) ≤ up c a + up c b := by
  have h : a + b ≤ c * (up c a + up c b) := by
    rw [Nat.mul_add]
    exact Nat.add_le_add (le_mul_up hc a) (le_mul_up hc b)
  simpa [up_mul hc] using up_mono c h

/-- A point maps into the interval `span` derives if it lies in the box `(lo, ext)` along the loops the expression mentions. -/
theorem span_sound_of_mentions (vals lo ext x : Nat → Nat) :
    ∀ e : AffE, e.WF → (∀ d, e.mentions d = true → x d < ext d) →
      (span vals lo ext e).2.1 ≤ eval vals (fun d => lo d + x d) e ∧
      eval vals (fun d => lo d + x d) e ≤ (span vals lo ext e).2.1 + (span vals lo ext e).2.2
  | dim d, _, hx => by have := hx d (by simp [mentions]); simp [span, eval]; omega
  | sym _, _, _ => by simp [span, eval]
  | lit _, _, _ => by simp [span, eval]
  | add a b, h, hx => by
    have ha := span_sound_of_mentions vals lo ext x a h.1 fun d hd => hx d (by simp [mentions, hd])
    have hb := span_sound_of_mentions vals lo ext x b h.2 fun d hd => hx d (by simp [mentions, hd])
    simp only [span, eval]
    omega
  | mul a c, h, hx => by
    have ha := span_sound_of_mentions vals lo ext x a h fun d hd => hx d (by simpa [mentions] using hd)
    simp only [span, eval]
    refine ⟨Nat.mul_le_mul_right c ha.1, ?_⟩
    rw [← Nat.add_mul]
    exact Nat.mul_le_mul_right c ha.2
  | fdiv a c, h, hx => by
    have ha := span_sound_of_mentions vals lo ext x a h.1 fun d hd => hx d (by simpa [mentions] using hd)
    simp only [span, eval]
    have h1 := Nat.div_le_div_right (c := c) ha.1
    have h2 := Nat.div_le_div_right (c := c) ha.2
    have h3 := Nat.div_le_div_right (c := c) (Nat.le_add_right (span vals lo ext a).2.1 (span vals lo ext a).2.2)
    omega
  | cdiv a c, h, hx => by
    have ha := span_sound_of_mentions vals lo ext x a h.1 fun d hd => hx d (by simpa [mentions] using hd)
    simp only [span, eval]
    have h1 := up_mono c ha.1
    have h2 := up_mono c ha.2
    have h3 := up_mono c (Nat.le_add_right (span vals lo ext a).2.1 (span vals lo ext a).2.2)
    omega
  | mod a c, h, hx => by
    have ha := span_sound_of_mentions vals lo ext x a h.1 fun d hd => hx d (by simpa [mentions] using hd)
    have hc := h.2
    simp only [span, eval]
    generalize eval vals (fun d => lo d + x d) a = v at ha ⊢
    generalize (span vals lo ext a).2.1 = l at ha ⊢
    generalize (span vals lo ext a).2.2 = w at ha ⊢
    have hv := Nat.mod_lt v hc
    split
    · simp; omega
    · rename_i hw
      simp only
      have hl := Nat.div_add_mod l c
      have e : v = l % c + (v - l) + c * (l / c) := by omega
      have hm : v % c = l % c + (v - l) :=
        calc v % c = (l % c + (v - l) + c * (l / c)) % c := congrArg (· % c) e
          _ = l % c + (v - l) := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (by omega)]
      omega

/-- Every point of the box `(lo, ext)` maps into the interval `span` derives. -/
theorem span_sound (vals lo ext x : Nat → Nat) (hx : ∀ d, x d < ext d) (e : AffE) (he : e.WF) :
    (span vals lo ext e).2.1 ≤ eval vals (fun d => lo d + x d) e ∧
    eval vals (fun d => lo d + x d) e ≤ (span vals lo ext e).2.1 + (span vals lo ext e).2.2 :=
  span_sound_of_mentions vals lo ext x e he fun d _ => hx d

theorem rel_zero (x : Nat → Nat) : ∀ e : AffE, (rel x e).1 = false → (rel x e).2 = 0
  | dim _, h => by simp [rel] at h
  | add a b, h => by
    have ha := rel_zero x a
    have hb := rel_zero x b
    revert h
    simp only [rel]
    rcases rel x a with ⟨_ | _, u⟩ <;> rcases rel x b with ⟨_ | _, v⟩ <;> simp
  | mul a c, h => by
    have ha := rel_zero x a h
    simp [rel, ha]
  | sym _, _ | lit _, _ | fdiv _ _, _ | cdiv _ _, _ | mod _ _, _ => rfl

theorem rel_add (x : Nat → Nat) (a b : AffE) : (rel x (add a b)).2 = (rel x a).2 + (rel x b).2 := by
  have ha := rel_zero x a
  have hb := rel_zero x b
  simp only [rel]
  cases hra : rel x a with | mk p u =>
  cases hrb : rel x b with | mk q v =>
  rw [hra] at ha; rw [hrb] at hb
  cases p <;> cases q <;> simp_all

/-- An affine expression at `lo + x` is the start of its interval plus its linear part at `x`, so a tile reads a staged box at `rel`. -/
theorem rel_sound (vals lo ext x : Nat → Nat) :
    ∀ e : AffE, (span vals lo ext e).1 = true →
      eval vals (fun d => lo d + x d) e = (span vals lo ext e).2.1 + (rel x e).2
  | dim _, _ => by simp [span, eval, rel]
  | sym _, _ => by simp [span, eval, rel]
  | lit _, _ => by simp [span, eval, rel]
  | add a b, h => by
    simp only [span, Bool.and_eq_true] at h
    have ha := rel_sound vals lo ext x a h.1
    have hb := rel_sound vals lo ext x b h.2
    rw [rel_add]
    simp only [span, eval]
    omega
  | mul a c, h => by
    have ha := rel_sound vals lo ext x a (by simpa [span] using h)
    simp only [span, eval, rel, ha, Nat.add_mul]
  | fdiv _ _, h | cdiv _ _, h => by simp [span] at h
  | mod _ _, h => by simp only [span] at h; split at h <;> simp at h

/-- The width of the interval stays below `reach` of any bound on the extents, wherever the box starts. -/
theorem span_le_reach (vals lo ext cap : Nat → Nat) (hcap : ∀ d, ext d ≤ cap d) :
    ∀ e : AffE, e.WF → (span vals lo ext e).2.2 ≤ reach cap e
  | dim d, _ => by have := hcap d; simp [span, reach]; omega
  | sym _, _ => by simp [span, reach]
  | lit _, _ => by simp [span, reach]
  | add a b, h => by
    have ha := span_le_reach vals lo ext cap hcap a h.1
    have hb := span_le_reach vals lo ext cap hcap b h.2
    simp only [span, reach]
    omega
  | mul a c, h => by
    have ha := span_le_reach vals lo ext cap hcap a h
    simp only [span, reach]
    exact Nat.mul_le_mul_right c ha
  | fdiv a c, h => by
    have ha := span_le_reach vals lo ext cap hcap a h.1
    have hc := h.2
    simp only [span, reach]
    generalize (span vals lo ext a).2.1 = l
    generalize (span vals lo ext a).2.2 = w at ha
    have h1 : (l + w) / c ≤ l / c + up c w := by
      have hl := Nat.div_add_mod l c
      have hm := Nat.mod_lt l hc
      have : l + w ≤ c * (l / c) + (w + (c - 1)) := by omega
      have := Nat.div_le_div_right (c := c) this
      rwa [Nat.mul_add_div hc] at this
    have := up_mono c ha
    omega
  | cdiv a c, h => by
    have ha := span_le_reach vals lo ext cap hcap a h.1
    simp only [span, reach]
    have h1 := up_add_le h.2 (span vals lo ext a).2.1 (span vals lo ext a).2.2
    have := up_mono c ha
    omega
  | mod a c, h => by
    have ha := span_le_reach vals lo ext cap hcap a h.1
    have hc := h.2
    simp only [span, reach]
    generalize (span vals lo ext a).2.1 = l
    generalize (span vals lo ext a).2.2 = w at ha
    have hm := Nat.mod_lt l hc
    split <;> split <;> simp <;> omega

end AffE

/-- `aff_box` of `tensor.mim` for one row: the start and the extent of what a tile reads. -/
def affBox (vals lo ext : Nat → Nat) (e : AffE) : Nat × Nat :=
  ((e.span vals lo ext).2.1, (e.span vals lo ext).2.2 + 1)

/-- `aff_box_sound`: a tile's reads of a row fall into the box, and an affine row reads the box at its linear part. -/
theorem affBox_sound (vals lo ext x : Nat → Nat) (hx : ∀ d, x d < ext d) (e : AffE) (he : e.WF) :
    (affBox vals lo ext e).1 ≤ e.eval vals (fun d => lo d + x d) ∧
    e.eval vals (fun d => lo d + x d) < (affBox vals lo ext e).1 + (affBox vals lo ext e).2 ∧
    ((e.span vals lo ext).1 = true →
      e.eval vals (fun d => lo d + x d) - (affBox vals lo ext e).1 = (e.rel x).2) := by
  have h := AffE.span_sound vals lo ext x hx e he
  refine ⟨h.1, by simp only [affBox]; omega, fun ha => ?_⟩
  have := AffE.rel_sound vals lo ext x e ha
  simp only [affBox]
  omega

end Sched
