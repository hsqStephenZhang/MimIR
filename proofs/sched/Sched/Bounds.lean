import Sched.Store
import Sched.Machine

/-!
# The reads of a leaf stay in the box its views were built for

A leaf reads a slot at the positions of its tile, and an edge tile at the positions up to the static `cap` of its output loops.
The theorem says that under `echeck` every such position lies in a panel of the view's home box, for which the buffer was allocated.
It joins the ghost state of the axes, the ghost state of the views, and the run.
-/

namespace Sched

variable {Val : Type}

/-- The leaf runs its kernel over the `cap` of its output loops. -/
def edgeAt (a : AState) (vdim : Nat) : Bool := dynAt a vdim || dynAt a (vdim + 1)

/-- Every slot that reads a run-time output loop is packed along it, in panels at least as wide as the tile's `cap`. -/
def Packed (ro : Nat) (a : AState) (gs : List Ghost) : Prop :=
  ∀ e < ro, ∀ x, a[e]? = some x → x.dyn = true → ∀ g ∈ gs, g.reads e = true → x.cap ≤ g.unit e

theorem stageView_ghost (lo sz : Vec) (v : View Val) :
    (stageView lo sz v).g = if v.g.ready then v.g else stageGhost v.g := by
  simp only [stageView]
  split <;> rfl

/-- `echeck` on ghost state covers every leaf of the run: an edge leaf finds the slots it over-reads packed. -/
theorem echeck_sound (junk : Vec → Vec → Val) {rn ro : Nat} : ∀ (t : Tree) (a : AState) (vs : List (View Val)) (b : Box),
    echeck ro t a (vs.map (·.g)) = true → Holds a b →
    ∀ lf ∈ runV junk rn t vs b, ∃ a', Holds a' lf.1 ∧ (edgeAt a' lf.2.2 = true → Packed ro a' (lf.2.1.map (·.g)))
  | .leaf vdim, a, vs, b, hc, h, lf, hl => by
    simp only [runV, List.mem_singleton] at hl
    subst hl
    refine ⟨a, h, fun he e hero x hx hdyn g hg hre => ?_⟩
    simp only [echeck, edgeAt] at hc he
    simp only [he, Bool.not_true, Bool.false_or, List.all_eq_true, List.mem_range] at hc
    have := hc e hero
    simp only [hx, hdyn, Bool.not_true, Bool.false_or, List.all_eq_true] at this
    have := this g hg
    simpa [hre] using this
  | .reduce _ _ body, a, vs, b, hc, h, lf, hl => echeck_sound junk body a vs b hc h lf hl
  | .alloc _ body, a, vs, b, hc, h, lf, hl => echeck_sound junk body a vs b hc h lf hl
  | .pack d w s dep body, a, vs, b, hc, h, lf, hl => by
    simp only [runV] at hl
    simp only [echeck] at hc
    split at hl
    · rename_i v hv
      simp only [List.getElem?_map, hv, Option.map_some] at hc
      refine echeck_sound junk body a _ b ?_ h lf hl
      rw [List.map_set]
      exact hc
    · simp at hl
  | .stage s body, a, vs, b, hc, h, lf, hl => by
    simp only [runV] at hl
    simp only [echeck] at hc
    split at hl
    · rename_i v hv
      simp only [List.getElem?_map, hv, Option.map_some] at hc
      refine echeck_sound junk body a _ b ?_ h lf hl
      rw [List.map_set, stageView_ghost]
      exact hc
    · simp at hl
  | .split d ts mode body, a, vs, b, hc, h, lf, hl => by
    simp only [runV] at hl
    simp only [echeck] at hc
    cases hbd : b[d]? with
    | none => simp [hbd] at hl
    | some tl0 =>
      obtain ⟨lo, n⟩ := tl0
      rw [hbd] at hl
      obtain ⟨hdb, hbd'⟩ := List.getElem?_eq_some_iff.1 hbd
      have hda : d < a.length := h.1 ▸ hdb
      rw [List.getElem?_eq_getElem hda] at hc
      simp only [List.all_eq_true] at hc
      obtain ⟨tl, htl, hl⟩ := List.mem_flatMap.1 hl
      have hxb : a[d].Holds (lo, n) := hbd' ▸ h.2 d hda hdb
      obtain ⟨y, hy, hytl⟩ := atiles_sound (mode := mode) hxb htl
      refine echeck_sound junk body (a.set d y) _ (b.set d tl) ?_ (h.set d hytl) lf hl
      rw [List.map_map]
      exact hc y hy
  | .switch d c small big, a, vs, b, hc, h, lf, hl => by
    simp only [runV] at hl
    simp only [echeck] at hc
    cases hbd : b[d]? with
    | none => simp [hbd] at hl
    | some tl0 =>
      obtain ⟨lo, n⟩ := tl0
      rw [hbd] at hl
      obtain ⟨hdb, hbd'⟩ := List.getElem?_eq_some_iff.1 hbd
      have hda : d < a.length := h.1 ▸ hdb
      rw [List.getElem?_eq_getElem hda] at hc
      have hxb : a[d].Holds (lo, n) := hbd' ▸ h.2 d hda hdb
      have hsmall : n < c → Holds (a.set d (asmall c a[d])) b := fun hn => by
        have : (asmall c a[d]).Holds (lo, n) := by
          refine ⟨by simp only [asmall]; have := hxb.1; simp only at this; omega, fun hd => ?_⟩
          have := hxb.2 hd
          simp only [asmall] at this ⊢
          omega
        have := h.set d this
        rwa [← hbd', List.set_getElem_self] at this
      simp only at hl
      by_cases hdyn : a[d].dyn = true
      · simp only [hdyn, ite_true, Bool.and_eq_true] at hc
        split at hl
        · rename_i hn
          exact echeck_sound junk small _ vs b hc.1 (hsmall hn) lf hl
        · exact echeck_sound junk big a vs b hc.2 h lf hl
      · have hf : a[d].dyn = false := by simpa using hdyn
        have hn := hxb.2 hf
        simp only at hn
        simp only [hf, Bool.false_eq_true, ite_false] at hc
        rw [← hn] at hc
        split at hl
        · rename_i hlt
          simp only [hlt, ite_true] at hc
          exact echeck_sound junk small _ vs b hc (hsmall hlt) lf hl
        · rename_i hlt
          simp only [hlt, ite_false] at hc
          exact echeck_sound junk big a vs b hc h lf hl

theorem div_lt_up {x u n : Nat} (hu : 0 < u) (h : x < n) : x / u < AffE.up u n :=
  Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le h (AffE.le_mul_up hu n))

/-- Every position a leaf reads lies in a panel of the box its view was built for.
An edge leaf reads up to the `cap` of its output loops; along a run-time one its slots are packed, and the tile stays in the panel it starts in. -/
theorem reads_in_home {rn ro : Nat} {spec : Vec → Val} {lf : Box} {v : View Val} {a' : AState} {vdim : Nat}
    {gs : List Ghost} (hok : v.Ok rn spec lf) (hr : v.g.ready = true) (hlen : lf.length = rn) (ha : Holds a' lf)
    (hv : v.g ∈ gs) (hp : edgeAt a' vdim = true → Packed ro a' gs) (o : Vec)
    (ho : ∀ e < rn, o e < if e < ro ∧ edgeAt a' vdim = true then capOf a' e else boxSz lf e)
    (hpos : ∀ e < ro, 0 < boxSz lf e) :
    ∀ e < rn, v.g.reads e = true → (v.q e * v.g.unit e + o e) / v.g.unit e < AffE.up (v.g.unit e) (v.hsz e) := by
  intro e he hre
  obtain ⟨_, h⟩ := hok
  simp only [hr, ite_true] at h
  obtain ⟨_, hb, _, hu⟩ := h
  have hbe := hb e he
  have hue := hu e
  have hoe := ho e he
  by_cases hcase : e < ro ∧ edgeAt a' vdim = true
  · simp only [hcase, and_self, ite_true] at hoe
    have hea : e < a'.length := by rw [ha.1, hlen]; exact he
    have hel : e < lf.length := hlen ▸ he
    have hx := ha.2 e hea hel
    have hcap : capOf a' e = a'[e].cap := by simp [capOf, List.getElem?_eq_getElem hea]
    have hsz : boxSz lf e = lf[e].2 := by simp [boxSz, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hel]
    rw [hcap] at hoe
    by_cases hdyn : a'[e].dyn = true
    · have hw := hp hcase.2 e hcase.1 a'[e] (List.getElem?_eq_getElem hea) hdyn v.g hv hre
      have hpe := hpos e hcase.1
      have hq : (v.q e * v.g.unit e + o e) / v.g.unit e = v.q e := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hue, Nat.div_eq_of_lt (by omega), Nat.zero_add]
      rw [hq]
      have hc := AffE.le_mul_up hue (v.hsz e)
      refine Nat.lt_of_mul_lt_mul_left (a := v.g.unit e) ?_
      rw [Nat.mul_comm]
      omega
    · have := hx.2 (by simpa using hdyn)
      exact div_lt_up hue (by omega)
  · simp only [hcase, ite_false] at hoe
    exact div_lt_up hue (by omega)

/-- The panels a `Pack` allocates hold the panels of its home box: `levels_pack_one` counts them by the `cap` of the packed loop, which bounds its extent. -/
theorem home_panels {w n cap : Nat} (h : n ≤ cap) : AffE.up w n ≤ AffE.up w cap := AffE.up_mono w h

end Sched
