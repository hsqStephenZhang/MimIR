import Sched.Machine
import Sched.Rfactor

/-!
# The paths of a `Leaf`

`exec_leaf` has three ways to compute a tile besides the plain kernel.
Each of them is shown to leave in the output what the plain kernel leaves there.
-/

namespace Sched

variable {β α : Type}

/-- The flag of `span` only tells whether the expression divides or takes a modulus. -/
theorem AffE.span_fst (vals lo ext lo' ext' : Nat → Nat) :
    ∀ e : AffE, (e.span vals lo ext).1 = (e.span vals lo' ext').1
  | dim _ | sym _ | lit _ => rfl
  | add a b => by simp only [span, AffE.span_fst vals lo ext lo' ext' a, AffE.span_fst vals lo ext lo' ext' b]
  | mul a _ => by simp only [span, AffE.span_fst vals lo ext lo' ext' a]
  | fdiv _ _ | cdiv _ _ => rfl
  | mod _ _ => by simp only [span]; split <;> split <;> rfl

/-- The `flat` read: a buffer shifted to where the tile's corner maps is read at the linear part of an affine row.
`aff_linear` tests the row at the box `(0, 1)`. -/
theorem AffE.flat_read (vals corner o : Nat → Nat) (e : AffE)
    (h : (e.span vals (fun _ => 0) (fun _ => 1)).1 = true) :
    e.eval vals (fun d => corner d + o d) = (e.span vals corner (fun _ => 1)).2.1 + (e.rel o).2 :=
  AffE.rel_sound vals corner (fun _ => 1) o e (by rw [AffE.span_fst vals corner _ (fun _ => 0) (fun _ => 1)]; exact h)

/-- The second box starts where the first one does and has at least its extents. -/
inductive Grows : Box → Box → Prop where
  | nil : Grows [] []
  | cons (lo : Nat) {n n' : Nat} {r r' : Box} : n ≤ n' → Grows r r' → Grows ((lo, n) :: r) ((lo, n') :: r')

/-- A box with larger extents holds the points of the smaller one. -/
theorem points_mono {b b' : Box} (h : Grows b b') : ∀ c ∈ points b, c ∈ points b' := by
  induction h with
  | nil => exact fun _ h => h
  | cons lo hn _ ih =>
    intro c hc
    obtain ⟨i, p, rfl, h1, h2, hp⟩ := mem_points_cons.1 hc
    exact mem_points_cons.2 ⟨i, p, rfl, h1, by omega, ih p hp⟩

/-- The edge tile of a run-time extent: the kernel runs on a scratch tile `b'` of the extents `cap`, and only the cells of the tile `b` are copied back. -/
def Op.leafEdge (op : Op β α) (ro : Nat) (rb b b' : Box) (dst : Point → α) : Point → α :=
  fun c => if c ∈ points (b.take ro) then op.leaf ro rb b' dst c else dst c

theorem Op.leafEdge_eq (op : Op β α) (ro : Nat) (rb b b' : Box) (dst : Point → α)
    (hout : Grows (b.take ro) (b'.take ro))
    (hred : b'.drop ro = b.drop ro) : op.leafEdge ro rb b b' dst = op.leaf ro rb b dst := by
  funext c
  simp only [Op.leafEdge, Op.leaf, fresh, last, hred]
  by_cases hc : c ∈ points (b.take ro)
  · simp only [hc, points_mono hout c hc, ↓reduceIte]
  · simp only [hc, ↓reduceIte]

/-- A `Leaf` under a `Reduce`: the cell's reduction runs in `w` lanes that a merge joins. -/
def Op.leafLanes (op : Op β α) (merge : α → α → α) (w : Nat) (key : Point → Nat) (ro : Nat) (rb b : Box)
    (dst : Point → α) : Point → α := fun c =>
  if c ∈ points (b.take ro) then
    (if last ro rb b then op.post else id)
      (op.redLanes merge w key (b.drop ro) c (if fresh ro rb b then op.init else dst c))
  else dst c

theorem Op.leafLanes_eq (op : Op β α) {merge : α → α → α} (h : Merge op.f op.init merge) (w : Nat) (key : Point → Nat)
    (ro : Nat) (rb b : Box) (dst : Point → α) (hk : ∀ p ∈ points (b.drop ro), key p < w) :
    op.leafLanes merge w key ro rb b dst = op.leaf ro rb b dst := by
  funext c
  simp only [Op.leafLanes, Op.leaf, op.redLanes_eq h w key _ c _ hk]

/-- A tree whose leaves pick any of the three paths computes what it computes with plain leaves. -/
theorem Op.exec_paths (op : Op β α) {merge : α → α → α} (h : Merge op.f op.init merge) (ro : Nat) (rb : Box)
    (t : Tree) (b : Box) (dst : Point → α) (lanes : Box → Option (Nat × (Point → Nat))) (scratch : Box → Option Box)
    (hl : ∀ lf ∈ run t b, ∀ wk, lanes lf = some wk → ∀ p ∈ points (lf.drop ro), wk.2 p < wk.1)
    (hs : ∀ lf ∈ run t b, ∀ b', scratch lf = some b' →
      Grows (lf.take ro) (b'.take ro) ∧ b'.drop ro = lf.drop ro) :
    (run t b).foldl (fun dst lf =>
        match lanes lf, scratch lf with
        | some wk, _ => op.leafLanes merge wk.1 wk.2 ro rb lf dst
        | none, some b' => op.leafEdge ro rb lf b' dst
        | none, none => op.leaf ro rb lf dst) dst =
      op.exec ro rb t b dst := by
  refine foldl_congr_mem (fun lf hlf s => ?_) dst
  cases hlan : lanes lf with
  | some wk => exact op.leafLanes_eq h wk.1 wk.2 ro rb lf s (hl lf hlf wk hlan)
  | none =>
    cases hsc : scratch lf with
    | some b' =>
      obtain ⟨h1, h2⟩ := hs lf hlf b' hsc
      exact op.leafEdge_eq ro rb lf b' s h1 h2
    | none => rfl

end Sched
