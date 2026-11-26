import Sched.Aff
import Sched.Ghost

/-!
# `levels_need`: the storage an `Alloc` reserves

An `Alloc` sizes the storage of its slot by an abstract interpretation of the subtree that only follows `cap`.
The theorem says that every `Pack` and `Stage` that takes this storage passes its check that the elements it may fill fit.
-/

namespace Sched

def capOf (a : AState) (e : Nat) : Nat := match a[e]? with | some x => x.cap | none => 0
def dynOf (a : AState) (e : Nat) : Bool := match a[e]? with | some x => x.dyn | none => false

def upd {γ : Type} (f : Nat → γ) (d : Nat) (x : γ) : Nat → γ := fun e => if e = d then x else f e

/-- The elements of the panels a `Pack` allocates: `w` lanes times the panels of loop `d`, one spare for a run-time extent, times the extents of the other loops its slot reads. -/
def panel (rn d w : Nat) (dep : List Bool) (caps : Nat → Nat) (dyn : Bool) : Nat :=
  w * (AffE.up w (caps d) + if dyn then 1 else 0) *
    ((List.range rn).map fun e => if e ≠ d ∧ dep.getD e false = true then caps e else 1).foldr (· * ·) 1

/-- `levels_need` without its saturation at `levels_unbounded`: `box` gives the elements of a `Stage` for given bounds, and `fills` tells whether a `Stage` of the slot fills anything. -/
def need (rn slot : Nat) (box : (Nat → Nat) → Nat) (fills : Bool) : Tree → (Nat → Nat) → (Nat → Bool) → Nat
  | .leaf _, _, _ => 0
  | .split d ts mode body, caps, dyn =>
    need rn slot box fills body (upd caps d (if dyn d then ts else min (caps d) ts))
      (if mode = .loop then upd dyn d true else dyn)
  | .reduce _ _ body, caps, dyn => need rn slot box fills body caps dyn
  | .pack d w s dep body, caps, dyn =>
    if s = slot then panel rn d w dep caps (dyn d) else need rn slot box fills body caps dyn
  | .stage s body, caps, dyn => if s = slot ∧ fills = true then box caps else need rn slot box fills body caps dyn
  | .alloc s body, caps, dyn => if s = slot then 0 else need rn slot box fills body caps dyn
  | .switch d c small big, caps, dyn =>
    max (need rn slot box fills small (upd caps d (min (caps d) (c - 1))) dyn) (need rn slot box fills big caps dyn)

/-- The checks of the outermost `Pack`s and `Stage`s of the slot against the `sn` elements an `Alloc` above them holds, on the ghost states the interpreter reaches. -/
def fits (rn slot : Nat) (box : (Nat → Nat) → Nat) (fills : Bool) (sn : Nat) : Tree → AState → Bool
  | .leaf _, _ => true
  | .split d ts mode body, a =>
    match a[d]? with
    | some x => (atiles ts mode x).all fun y => fits rn slot box fills sn body (a.set d y)
    | none => true
  | .reduce _ _ body, a => fits rn slot box fills sn body a
  | .pack d w s dep body, a =>
    if s = slot then decide (panel rn d w dep (capOf a) (dynOf a d) ≤ sn) else fits rn slot box fills sn body a
  | .stage s body, a =>
    if s = slot ∧ fills = true then decide (box (capOf a) ≤ sn) else fits rn slot box fills sn body a
  | .alloc _ body, a => fits rn slot box fills sn body a
  | .switch d c small big, a =>
    match a[d]? with
    | some x =>
      if x.dyn then fits rn slot box fills sn small (a.set d (asmall c x)) && fits rn slot box fills sn big a
      else if x.cap < c then fits rn slot box fills sn small (a.set d (asmall c x)) else fits rn slot box fills sn big a
    | none => true

/-- No `Alloc` of the slot lies in the tree, so the storage of the one above reaches every user. -/
def Tree.allocFree (slot : Nat) : Tree → Bool
  | .leaf _ => true
  | .split _ _ _ body | .reduce _ _ body | .pack _ _ _ _ body | .stage _ body => body.allocFree slot
  | .alloc s body => s ≠ slot && body.allocFree slot
  | .switch _ _ small big => small.allocFree slot && big.allocFree slot

theorem foldr_mul_le : ∀ (l : List Nat) (f g : Nat → Nat), (∀ e, f e ≤ g e) →
    (l.map f).foldr (· * ·) 1 ≤ (l.map g).foldr (· * ·) 1
  | [], _, _, _ => Nat.le_refl _
  | e :: l, f, g, h => by
    simp only [List.map_cons, List.foldr_cons]
    exact Nat.mul_le_mul (h e) (foldr_mul_le l f g h)

theorem panel_mono (rn d w : Nat) (dep : List Bool) {caps caps' : Nat → Nat} {dyn dyn' : Bool}
    (hc : ∀ e, caps e ≤ caps' e) (hd : dyn = true → dyn' = true) :
    panel rn d w dep caps dyn ≤ panel rn d w dep caps' dyn' := by
  unfold panel
  refine Nat.mul_le_mul (Nat.mul_le_mul_left _ (Nat.add_le_add (AffE.up_mono w (hc d)) ?_)) (foldr_mul_le _ _ _ fun e => ?_)
  · cases dyn <;> cases dyn' <;> simp_all
  · split
    · exact hc e
    · exact Nat.le_refl _

theorem capOf_set {a : AState} {d : Nat} (hd : d < a.length) (y : AAxis) (e : Nat) :
    capOf (a.set d y) e = if e = d then y.cap else capOf a e := by
  simp only [capOf, List.getElem?_set]
  by_cases h : d = e
  · subst h; simp [hd]
  · have : ¬e = d := fun h' => h h'.symm
    simp [h, this]

theorem dynOf_set {a : AState} {d : Nat} (hd : d < a.length) (y : AAxis) (e : Nat) :
    dynOf (a.set d y) e = if e = d then y.dyn else dynOf a e := by
  simp only [dynOf, List.getElem?_set]
  by_cases h : d = e
  · subst h; simp [hd]
  · have : ¬e = d := fun h' => h h'.symm
    simp [h, this]

/-- What `levels_need` assumes of the tiles of a `Split` covers every state `split_axis` runs its body in. -/
theorem atiles_need {ts : Nat} {mode : Mode} {x y : AAxis} (hts : 0 < ts) (hy : y ∈ atiles ts mode x) {cap : Nat}
    {dyn : Bool} (hc : x.cap ≤ cap) (hd : x.dyn = true → dyn = true) :
    y.cap ≤ (if dyn then ts else min cap ts) ∧ (y.dyn = true → (if mode = .loop then true else dyn) = true) := by
  have hm := Nat.mod_lt x.cap hts
  have hml := Nat.mod_le x.cap ts
  simp only [atiles] at hy
  by_cases hx : x.dyn = true
  · have hdt := hd hx
    simp only [hx, ite_true] at hy
    split at hy
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
      rcases hy with rfl | rfl <;> simp [hdt] <;> omega
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
      subst hy
      simp [hdt]
      omega
  · have hx' : x.dyn = false := by simpa using hx
    simp only [hx', Bool.false_eq_true, ite_false] at hy
    split at hy
    · rename_i hl
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
      subst hy
      simp only [hl.1, ite_true]
      cases dyn <;> simp <;> omega
    · rcases List.mem_append.1 hy with hy | hy
      · split at hy
        · rename_i hdiv
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
          subst hy
          have : ts ≤ x.cap := by
            refine Nat.le_of_not_lt fun hlt => hdiv (Nat.div_eq_of_lt hlt)
          simp only [Bool.false_eq_true, false_imp_iff, and_true]
          cases dyn <;> simp <;> omega
        · simp at hy
      · split at hy
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
          subst hy
          simp only [Bool.false_eq_true, false_imp_iff, and_true]
          cases dyn <;> simp <;> omega
        · simp at hy

/-- The storage `levels_need` computes passes the checks of every `Pack` and `Stage` that takes it. -/
theorem need_fits {rn ro slot : Nat} (box : (Nat → Nat) → Nat) (fills : Bool)
    (hbox : ∀ c c' : Nat → Nat, (∀ e, c e ≤ c' e) → box c ≤ box c') :
    ∀ (t : Tree) (a : AState) (caps : Nat → Nat) (dyn : Nat → Bool) (sn : Nat), t.wf rn ro = true →
      t.allocFree slot = true → (∀ e, capOf a e ≤ caps e) → (∀ e, dynOf a e = true → dyn e = true) →
      need rn slot box fills t caps dyn ≤ sn → fits rn slot box fills sn t a = true
  | .leaf _, _, _, _, _, _, _, _, _, _ => rfl
  | .reduce _ _ body, a, caps, dyn, sn, hw, hf, hc, hd, hn => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    exact need_fits box fills hbox body a caps dyn sn hw.2 hf hc hd hn
  | .alloc s body, a, caps, dyn, sn, hw, hf, hc, hd, hn => by
    simp only [Tree.allocFree, Bool.and_eq_true, decide_eq_true_eq] at hf
    simp only [need, hf.1, ite_false] at hn
    exact need_fits box fills hbox body a caps dyn sn hw hf.2 hc hd hn
  | .pack d w s dep body, a, caps, dyn, sn, hw, hf, hc, hd, hn => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    simp only [need] at hn
    simp only [fits]
    split
    · rename_i hs
      simp only [hs, ite_true] at hn
      exact decide_eq_true (Nat.le_trans (panel_mono rn d w dep hc (hd d)) hn)
    · rename_i hs
      simp only [hs, ite_false] at hn
      exact need_fits box fills hbox body a caps dyn sn hw.2 hf hc hd hn
  | .stage s body, a, caps, dyn, sn, hw, hf, hc, hd, hn => by
    simp only [need] at hn
    simp only [fits]
    split
    · rename_i hs
      simp only [hs, and_self, ite_true] at hn
      exact decide_eq_true (Nat.le_trans (hbox _ _ hc) hn)
    · rename_i hs
      simp only [hs, ite_false] at hn
      exact need_fits box fills hbox body a caps dyn sn hw hf hc hd hn
  | .switch d c small big, a, caps, dyn, sn, hw, hf, hc, hd, hn => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    simp only [Tree.allocFree, Bool.and_eq_true] at hf
    simp only [need] at hn
    have hn1 := Nat.le_trans (Nat.le_max_left _ _) hn
    have hn2 := Nat.le_trans (Nat.le_max_right _ _) hn
    simp only [fits]
    split
    · rename_i x hx
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hs := need_fits box fills hbox small (a.set d (asmall c x)) _ dyn sn (by simpa using hw.1.2) hf.1
        (fun e => by
          rw [capOf_set hda]
          simp only [upd]
          split
          · rename_i he
            subst he
            have := hc e
            simp only [capOf, hx] at this
            simp only [asmall]
            omega
          · exact hc e)
        (fun e => by
          rw [dynOf_set hda]
          split
          · rename_i he
            subst he
            have := hd e
            simpa only [dynOf, hx, asmall] using this
          · exact hd e) hn1
      have hb := need_fits box fills hbox big a caps dyn sn hw.2 hf.2 hc hd hn2
      split
      · simp [hs, hb]
      · split
        · exact hs
        · exact hb
    · rfl
  | .split d ts mode body, a, caps, dyn, sn, hw, hf, hc, hd, hn => by
    simp only [Tree.wf, Bool.and_eq_true, decide_eq_true_eq] at hw
    simp only [need] at hn
    simp only [fits]
    split
    · rename_i x hx
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      refine List.all_eq_true.2 fun y hy => ?_
      have hcx : x.cap ≤ caps d := by
        have := hc d
        simpa only [capOf, hx] using this
      have hdx : x.dyn = true → dyn d = true := by
        have := hd d
        simpa only [dynOf, hx] using this
      obtain ⟨h1, h2⟩ := atiles_need (mode := mode) hw.1.2 hy hcx hdx
      refine need_fits box fills hbox body (a.set d y) _ _ sn (by simpa using hw.2) hf (fun e => ?_) (fun e => ?_) hn
      · rw [capOf_set hda]
        simp only [upd]
        split
        · rename_i he
          subst he
          exact h1
        · exact hc e
      · rw [dynOf_set hda]
        split
        · rename_i he
          subst he
          intro hyd
          have := h2 hyd
          split
          · simp [upd]
          · rename_i hm
            simpa [hm] using this
        · intro hde
          have := hd e hde
          split
          · rename_i he _
            simp [upd, he, this]
          · exact this
    · rfl

end Sched
