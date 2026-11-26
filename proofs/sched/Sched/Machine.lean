import Sched.Exact
import Sched.View

/-!
# The machine: a tree over views

The run of a tree whose leaves read their inputs through the views that `Split`, `Pack` and `Stage` hand them.
The theorem joins the two halves: the views supply the specified inputs, and the boxes add them up in the specified order.
Its corollaries are the entry with an empty run-time reduction, and the premise a `Stage` needs of a producer that is itself such a run.
-/

namespace Sched

variable {Val α : Type}

/-- A combiner over the values of all slots. -/
structure MOp (Val α : Type) where
  f    : α → List Val → α
  init : α
  post : α → α

def MOp.on (m : MOp Val α) (inp : Point → List Val) : Op (List Val) α := ⟨inp, m.f, m.init, m.post⟩

/-- What a leaf on the box `lf` reads at the point `p` through its views. -/
def viewInp (lf : Box) (vs : List (View Val)) : Point → List Val :=
  fun p => vs.map fun v => v.rd v.q fun e => vecOf p e - boxLo lf e

/-- What the slots specify at the point `p`. -/
def specInp (specs : List (Vec → Val)) : Point → List Val := fun p => specs.map (· (vecOf p))

def MOp.execV (m : MOp Val α) (junk : Vec → Vec → Val) (rn ro : Nat) (rb : Box) (t : Tree) (vs : List (View Val))
    (b : Box) (dst : Point → α) : Point → α :=
  (runV junk rn t vs b).foldl (fun dst lf => (m.on (viewInp lf.1 lf.2.1)).leaf ro rb lf.1 dst) dst

theorem runV_fst (junk : Vec → Vec → Val) (rn : Nat) : ∀ (t : Tree) (vs : List (View Val)) (b : Box),
    vcheck rn t (vs.map (·.g)) = true → (runV junk rn t vs b).map (·.1) = run t b
  | .leaf _, vs, b, _ => rfl
  | .reduce _ _ body, vs, b, hc => runV_fst junk rn body vs b hc
  | .alloc _ body, vs, b, hc => runV_fst junk rn body vs b hc
  | .switch d c small big, vs, b, hc => by
    simp only [vcheck, Bool.and_eq_true] at hc
    simp only [runV, run]
    cases b[d]? with
    | none => rfl
    | some tl =>
      obtain ⟨lo, n⟩ := tl
      simp only
      split
      · exact runV_fst junk rn small vs b hc.1
      · exact runV_fst junk rn big vs b hc.2
  | .split d ts mode body, vs, b, hc => by
    simp only [vcheck, Bool.and_eq_true] at hc
    simp only [runV, run]
    cases b[d]? with
    | none => rfl
    | some tl =>
      obtain ⟨lo, n⟩ := tl
      simp only
      rw [List.map_flatMap]
      congr 1
      funext tl
      refine runV_fst junk rn body _ _ ?_
      rw [List.map_map]
      exact hc.2
  | .pack d w slot dep body, vs, b, hc => by
    simp only [vcheck] at hc
    simp only [runV, run]
    split
    · rename_i v hv
      simp only [List.getElem?_map, hv, Option.map_some, Bool.and_eq_true] at hc
      refine runV_fst junk rn body _ b ?_
      rw [List.map_set]
      exact hc.2
    · rename_i hv
      simp [List.getElem?_map, hv] at hc
  | .stage slot body, vs, b, hc => by
    simp only [vcheck] at hc
    simp only [runV, run]
    split
    · rename_i v hv
      simp only [List.getElem?_map, hv, Option.map_some] at hc
      refine runV_fst junk rn body _ b ?_
      rw [List.map_set]
      have : (stageView (boxLo b) (boxSz b) v).g = if v.g.ready then v.g else stageGhost v.g := by
        simp only [stageView]
        split <;> rfl
      rw [this]
      exact hc
    · rename_i hv
      simp [List.getElem?_map, hv] at hc

theorem mem_points_append : ∀ {b₁ b₂ : Box} {c r : Point}, c ∈ points b₁ → r ∈ points b₂ → c ++ r ∈ points (b₁ ++ b₂)
  | [], b₂, c, r, hc, hr => by
    simp only [points, List.mem_singleton] at hc
    subst hc
    simpa using hr
  | (lo, n) :: b₁, b₂, c, r, hc, hr => by
    obtain ⟨i, p, rfl, h1, h2, hp⟩ := mem_points_cons.1 hc
    exact mem_points_cons.2 ⟨i, p ++ r, rfl, h1, h2, mem_points_append hp hr⟩

/-- A leaf only reads its inputs at the points of its box. -/
theorem Op.leaf_congr {β : Type} (op op' : Op β α) (ro : Nat) (rb b : Box) (dst : Point → α) (hf : op.f = op'.f)
    (hi : op.init = op'.init) (hp : op.post = op'.post) (h : ∀ p ∈ points b, op.inp p = op'.inp p) :
    op.leaf ro rb b dst = op'.leaf ro rb b dst := by
  funext c
  simp only [Op.leaf, hi, hp]
  split
  · rename_i hc
    refine congrArg _ ?_
    simp only [Op.red, hf]
    refine foldl_congr_mem (fun r hr a => ?_) _
    rw [h (c ++ r) (by simpa using mem_points_append hc hr)]
  · rfl

/-- The machine computes what the tree computes on the specified inputs. -/
theorem MOp.execV_eq (m : MOp Val α) (junk : Vec → Vec → Val) {rn ro : Nat} (specs : List (Vec → Val)) (rb : Box)
    (t : Tree) (vs : List (View Val)) (b : Box) (dst : Point → α) (hw : t.wf rn ro = true)
    (hc : vcheck rn t (vs.map (·.g)) = true) (hb : b.length = rn) (h : AllOk rn specs b vs) :
    m.execV junk rn ro rb t vs b dst = (m.on (specInp specs)).exec ro rb t b dst := by
  have hleaf : ∀ lf ∈ runV junk rn t vs b, ∀ s : Point → α,
      (m.on (viewInp lf.1 lf.2.1)).leaf ro rb lf.1 s = (m.on (specInp specs)).leaf ro rb lf.1 s := by
    intro lf hlf s
    obtain ⟨hok, hready⟩ := view_sound junk specs t vs b hw hc hb h lf hlf
    have hlen : lf.1.length = rn := by
      have : lf.1 ∈ run t b := by
        rw [← runV_fst junk rn t vs b hc]
        exact List.mem_map_of_mem hlf
      rw [length_of_mem_run this, hb]
    refine Op.leaf_congr _ _ ro rb lf.1 s rfl rfl rfl fun p hp => ?_
    simp only [MOp.on, viewInp, specInp]
    refine List.ext_getElem (by simpa using hok.1) fun i h1 h2 => ?_
    simp only [List.getElem_map]
    have h1' : i < lf.2.1.length := by simpa using h1
    exact (hok.2 i h1' (by simpa using h2)).read (hready _ (List.getElem_mem h1')) hlen hp
  rw [MOp.execV, foldl_congr_mem hleaf, Op.exec, ← runV_fst junk rn t vs b hc, List.foldl_map]

/-- `run_levels_at`: a run-time reduction extent of 0 fills the output with `init` and applies `post`, and anything else runs the tree. -/
def MOp.entry (m : MOp Val α) (junk : Vec → Vec → Val) (rn ro : Nat) (redDyn : Bool) (t : Tree) (vs : List (View Val))
    (b : Box) (dst : Point → α) : Point → α :=
  if redDyn = true ∧ (b.drop ro).any (fun tl => tl.2 == 0) = true then
    fun c => if c ∈ points (b.take ro) then m.post m.init else dst c
  else m.execV junk rn ro b t vs b dst

/-- The main theorem: from an uninitialized output buffer, a tree whose checks pass computes `post` of each cell's left fold over the specified inputs. -/
theorem MOp.entry_spec (m : MOp Val α) (junk : Vec → Vec → Val) {rn ro lanes : Nat} (redDyn : Bool)
    (specs : List (Vec → Val)) (t : Tree) (a : AState) (vs : List (View Val)) (b : Box) (dst : Point → α)
    (hc : check lanes ro t a = true) (hv : vcheck rn t (vs.map (·.g)) = true) (ha : Holds a b) (hb : b.length = rn)
    (hok : AllOk rn specs b vs) (hpos : redDyn = true ∨ ∀ tl ∈ b.drop ro, 0 < tl.2)
    (hl : lanes = 0 ∨ (m.on (specInp specs)).Comm) :
    m.entry junk rn ro redDyn t vs b dst =
      fun c => if c ∈ points (b.take ro) then m.post ((m.on (specInp specs)).red (b.drop ro) c m.init) else dst c := by
  unfold MOp.entry
  split
  · rename_i he
    obtain ⟨tl, htl, h0⟩ := List.any_eq_true.1 he.2
    have : points (b.drop ro) = [] := points_eq_nil_iff.2 ⟨tl, htl, by simpa using h0⟩
    funext c
    simp [Op.red, this]
  · rename_i he
    have hpos : ∀ tl ∈ b.drop ro, 0 < tl.2 := by
      intro tl htl
      rcases hpos with hd | hpos
      · refine Nat.pos_of_ne_zero fun h0 => he ⟨hd, List.any_eq_true.2 ⟨tl, htl, by simpa using h0⟩⟩
      · exact hpos tl htl
    have hw : t.wf rn ro = true := by
      have := check_wf hc
      rwa [ha.1, hb] at this
    rw [m.execV_eq junk specs b t vs b dst hw hv hb hok]
    exact (m.on (specInp specs)).exec_spec t a b dst hc ha hpos hl

/-- A producer that is itself a run over the box it is asked for meets the premise of `Stage`:
its buffer holds, at `y`, `post` of the fold of the cell `blo + y`. -/
theorem MOp.fill_spec (m : MOp Val α) (junk : Vec → Vec → Val) {rn ro lanes : Nat} (specs : List (Vec → Val))
    (red : Box) (t : Box → Tree) (a : Box → AState) (vs : Box → List (View Val)) (dst : Point → α)
    (cells : Vec → Vec → Box) (cell : Vec → Point)
    (hcell : ∀ blo bext y : Vec, (∀ i, y i < bext i) → cell (fun i => blo i + y i) ∈ points (cells blo bext))
    (hlen : ∀ blo bext, (cells blo bext).length = ro)
    (hrun : ∀ blo bext, let b := cells blo bext ++ red
      check lanes ro (t b) (a b) = true ∧ vcheck rn (t b) ((vs b).map (·.g)) = true ∧ Holds (a b) b ∧ b.length = rn ∧
        AllOk rn specs b (vs b))
    (hpos : ∀ tl ∈ red, 0 < tl.2) (hl : lanes = 0 ∨ (m.on (specInp specs)).Comm) (blo bext y : Vec)
    (hy : ∀ i, y i < bext i) :
    m.entry junk rn ro false (t (cells blo bext ++ red)) (vs (cells blo bext ++ red)) (cells blo bext ++ red) dst
        (cell fun i => blo i + y i) =
      m.post ((m.on (specInp specs)).red red (cell fun i => blo i + y i) m.init) := by
  obtain ⟨hc, hv, ha, hb, hok⟩ := hrun blo bext
  have hdrop : (cells blo bext ++ red).drop ro = red := by
    have hle : ro ≤ (cells blo bext).length := Nat.le_of_eq (hlen blo bext).symm
    rw [List.drop_append_of_le_length hle, List.drop_eq_nil_of_le (Nat.le_of_eq (hlen blo bext))]
    rfl
  have htake : (cells blo bext ++ red).take ro = cells blo bext := by
    have hle : ro ≤ (cells blo bext).length := Nat.le_of_eq (hlen blo bext).symm
    rw [List.take_append_of_le_length hle, List.take_of_length_le (Nat.le_of_eq (hlen blo bext))]
  rw [m.entry_spec junk false specs _ _ _ _ dst hc hv ha hb hok (.inr (by rw [hdrop]; exact hpos)) hl]
  simp only [hdrop, htake, hcell blo bext y hy, ite_true]

end Sched
