import Sched.Ghost

/-!
# A tree computes what one leaf computes

The value semantics of a leaf follows `exec_leaf`: the output buffer starts out uninitialized, the leaf that starts a cell's reduction ignores what the buffer holds, and the leaf that ends it applies `post`.
The theorem says that a tree whose checks pass leaves in the buffer what a single `Leaf` over the whole box leaves there.
A combiner without a merge keeps the order of every cell's additions, so the result is exact for floating point.
-/

namespace Sched

/-- A map-reduce whose output cell is the first `ro` coordinates of a point. -/
structure Op (β α : Type) where
  inp  : Point → β
  f    : α → β → α
  init : α
  post : α → α

variable {β α : Type}

/-- The order of a cell's additions does not matter. -/
def Op.Comm (op : Op β α) : Prop := ∀ a x y, op.f (op.f a x) y = op.f (op.f a y) x

/-- The box starts every reduction loop where the root box does. -/
def fresh (ro : Nat) (rb b : Box) : Prop := (b.drop ro).map Prod.fst = (rb.drop ro).map Prod.fst

/-- The box ends every reduction loop where the root box does. -/
def last (ro : Nat) (rb b : Box) : Prop :=
  (b.drop ro).map (fun tl => tl.1 + tl.2) = (rb.drop ro).map (fun tl => tl.1 + tl.2)

instance (ro : Nat) (rb b : Box) : Decidable (fresh ro rb b) := by unfold fresh; infer_instance
instance (ro : Nat) (rb b : Box) : Decidable (last ro rb b) := by unfold last; infer_instance

/-- The additions of the reduction box `r` to the cell `c`, in lexicographic order. -/
def Op.red (op : Op β α) (r : Box) (c : Point) (a : α) : α :=
  (points r).foldl (fun a p => op.f a (op.inp (c ++ p))) a

/-- What a `Leaf` on the box `b` leaves in the output buffer `dst`. -/
def Op.leaf (op : Op β α) (ro : Nat) (rb b : Box) (dst : Point → α) : Point → α := fun c =>
  if c ∈ points (b.take ro) then
    (if last ro rb b then op.post else id) (op.red (b.drop ro) c (if fresh ro rb b then op.init else dst c))
  else dst c

def Op.exec (op : Op β α) (ro : Nat) (rb : Box) (t : Tree) (b : Box) (dst : Point → α) : Point → α :=
  (run t b).foldl (fun dst lf => op.leaf ro rb lf dst) dst

/-- The box lies within the root box along the reduction loops and is not empty there. -/
def Sub (ro : Nat) (rb b : Box) : Prop :=
  b.length = rb.length ∧ ∀ d, ro ≤ d → ∀ (h : d < b.length) (h' : d < rb.length),
    rb[d].1 ≤ b[d].1 ∧ b[d].1 + b[d].2 ≤ rb[d].1 + rb[d].2 ∧ 0 < b[d].2

theorem Sub.set {ro : Nat} {rb b : Box} (h : Sub ro rb b) (d : Nat) (tl : Tile)
    (htl : ro ≤ d → ∀ h' : d < rb.length, rb[d].1 ≤ tl.1 ∧ tl.1 + tl.2 ≤ rb[d].1 + rb[d].2 ∧ 0 < tl.2) :
    Sub ro rb (b.set d tl) := by
  refine ⟨by simpa using h.1, fun e he hb hr => ?_⟩
  simp only [List.getElem_set]
  split
  · rename_i hde
    subst hde
    exact htl he hr
  · exact h.2 e he (by simpa using hb) hr

theorem foldl_congr_mem {σ τ : Type} {l : List τ} {f g : σ → τ → σ} (h : ∀ x ∈ l, ∀ s, f s x = g s x) (s : σ) :
    l.foldl f s = l.foldl g s := by
  induction l generalizing s with
  | nil => rfl
  | cons x l ih =>
    simp only [List.foldl_cons]
    rw [h x (by simp), ih fun y hy => h y (by simp [hy])]

theorem foldl_id {σ τ : Type} (l : List τ) (s : σ) : l.foldl (fun s _ => s) s = s := by
  induction l with
  | nil => rfl
  | cons _ _ ih => simpa using ih

theorem mem_points_set_split (lo m k : Nat) : ∀ (b : Box) (d : Nat) (c : Point), d < b.length →
    (c ∈ points (b.set d (lo, m + k)) ↔ c ∈ points (b.set d (lo, m)) ∨ c ∈ points (b.set d (lo + m, k))) ∧
    ¬(c ∈ points (b.set d (lo, m)) ∧ c ∈ points (b.set d (lo + m, k)))
  | [], _, _, h => by simp at h
  | (l0, n0) :: r, 0, c, _ => by
    simp only [List.set_cons_zero, mem_points_cons]
    refine ⟨⟨?_, ?_⟩, ?_⟩
    · rintro ⟨i, p, rfl, h1, h2, hp⟩
      by_cases hi : i < lo + m
      · exact .inl ⟨i, p, rfl, h1, hi, hp⟩
      · exact .inr ⟨i, p, rfl, by omega, by omega, hp⟩
    · rintro (⟨i, p, rfl, h1, h2, hp⟩ | ⟨i, p, rfl, h1, h2, hp⟩)
      · exact ⟨i, p, rfl, h1, by omega, hp⟩
      · exact ⟨i, p, rfl, by omega, by omega, hp⟩
    · rintro ⟨⟨i, p, rfl, _, h2, _⟩, ⟨j, q, he, h3, _, _⟩⟩
      injection he with he _
      omega
  | (l0, n0) :: r, d + 1, c, h => by
    have ih := fun p => mem_points_set_split lo m k r d p (by simpa using h)
    simp only [List.set_cons_succ, mem_points_cons]
    refine ⟨⟨?_, ?_⟩, ?_⟩
    · rintro ⟨i, p, rfl, h1, h2, hp⟩
      rcases (ih p).1.1 hp with hp | hp
      · exact .inl ⟨i, p, rfl, h1, h2, hp⟩
      · exact .inr ⟨i, p, rfl, h1, h2, hp⟩
    · rintro (⟨i, p, rfl, h1, h2, hp⟩ | ⟨i, p, rfl, h1, h2, hp⟩)
      · exact ⟨i, p, rfl, h1, h2, (ih p).1.2 (.inl hp)⟩
      · exact ⟨i, p, rfl, h1, h2, (ih p).1.2 (.inr hp)⟩
    · rintro ⟨⟨i, p, rfl, _, _, hp⟩, ⟨j, q, he, _, _, hq⟩⟩
      injection he with _ he
      subst he
      exact (ih p).2 ⟨hp, hq⟩

theorem Op.leaf_of_nil (op : Op β α) (ro : Nat) (rb b : Box) (dst : Point → α) (h : points (b.take ro) = []) :
    op.leaf ro rb b dst = dst := by
  funext c
  simp [Op.leaf, h]

/-- Two adjacent tiles of an output loop compute what the tile they make up computes. -/
theorem Op.leaf_seq_out (op : Op β α) {ro d : Nat} (rb b : Box) (lo m k : Nat) (dst : Point → α)
    (hd : d < ro) (hdb : d < b.length) :
    op.leaf ro rb (b.set d (lo + m, k)) (op.leaf ro rb (b.set d (lo, m)) dst) = op.leaf ro rb (b.set d (lo, m + k)) dst := by
  funext c
  have hdrop : ∀ tl, (b.set d tl).drop ro = b.drop ro := fun tl => List.drop_set_of_lt hd
  obtain ⟨hiff, hdis⟩ := mem_points_set_split lo m k (b.take ro) d c (by simp; omega)
  simp only [Op.leaf, fresh, last, hdrop, List.take_set]
  by_cases h1 : c ∈ points ((b.take ro).set d (lo, m))
  · have h2 : c ∉ points ((b.take ro).set d (lo + m, k)) := fun h2 => hdis ⟨h1, h2⟩
    have hw := hiff.2 (.inl h1)
    simp only [h1, h2, hw, ↓reduceIte]
  · by_cases h2 : c ∈ points ((b.take ro).set d (lo + m, k))
    · have hw := hiff.2 (.inr h2)
      simp only [h1, h2, hw, ↓reduceIte]
    · have hw : c ∉ points ((b.take ro).set d (lo, m + k)) := fun hw => (hiff.1 hw).elim h1 h2
      simp only [h1, h2, hw, ↓reduceIte]

theorem points_cons_append (lo m k : Nat) (r : Box) :
    points ((lo, m + k) :: r) = points ((lo, m) :: r) ++ points ((lo + m, k) :: r) := by
  simp only [points, ← List.flatMap_append, List.range'_append_1]

theorem map_set_getElem {l : List Tile} {r : List Nat} {g : Tile → Nat} {j : Nat} {x : Tile}
    (h : (l.set j x).map g = r) (_hj : j < l.length) (hr : j < r.length) : r[j] = g x := by
  subst h
  simp

/-- Two adjacent tiles of a reduction loop compute what the tile they make up computes: exactly for the first reduction loop, and up to the order of the additions for the others. -/
theorem Op.leaf_seq_red (op : Op β α) {ro : Nat} (rb b : Box) (j lo m k : Nat) (dst : Point → α)
    (hdb : ro + j < b.length) (hrb : ro + j < rb.length) (hl : rb[ro + j].1 ≤ lo)
    (hr : lo + m + k ≤ rb[ro + j].1 + rb[ro + j].2) (hm : 0 < m) (hk : 0 < k) (hc : j = 0 ∨ op.Comm) :
    op.leaf ro rb (b.set (ro + j) (lo + m, k)) (op.leaf ro rb (b.set (ro + j) (lo, m)) dst) =
      op.leaf ro rb (b.set (ro + j) (lo, m + k)) dst := by
  funext c
  have htake : ∀ tl, (b.set (ro + j) tl).take ro = b.take ro := fun tl => List.take_set_of_le (Nat.le_add_right ..)
  have hdrop : ∀ tl, (b.set (ro + j) tl).drop ro = (b.drop ro).set j tl := fun tl => by
    rw [List.drop_set, ite_eq_right_of_eq_false _ _ (eq_false (by omega)), Nat.add_sub_cancel_left]
  have hjd : j < (b.drop ro).length := by simp; omega
  have hjr : j < (rb.drop ro).length := by simp; omega
  have hf2 : ¬fresh ro rb (b.set (ro + j) (lo + m, k)) := fun h => by
    have h' : ((b.drop ro).set j (lo + m, k)).map Prod.fst = (rb.drop ro).map Prod.fst := by
      simpa only [fresh, hdrop] using h
    have := map_set_getElem h' hjd (by simpa using hjr)
    simp at this
    omega
  have hl1 : ¬last ro rb (b.set (ro + j) (lo, m)) := fun h => by
    have h' : ((b.drop ro).set j (lo, m)).map (fun tl => tl.1 + tl.2) = (rb.drop ro).map (fun tl => tl.1 + tl.2) := by
      simpa only [last, hdrop] using h
    have := map_set_getElem h' hjd (by simpa using hjr)
    simp at this
    omega
  have hf : fresh ro rb (b.set (ro + j) (lo, m)) = fresh ro rb (b.set (ro + j) (lo, m + k)) := by
    simp only [fresh, hdrop, List.map_set]
  have hla : last ro rb (b.set (ro + j) (lo + m, k)) = last ro rb (b.set (ro + j) (lo, m + k)) := by
    simp only [last, hdrop, List.map_set, Nat.add_assoc]
  have hred : ∀ a, op.red ((b.drop ro).set j (lo, m + k)) c a =
      op.red ((b.drop ro).set j (lo + m, k)) c (op.red ((b.drop ro).set j (lo, m)) c a) := fun a => by
    simp only [Op.red, ← List.foldl_append]
    rcases hc with rfl | hc
    · rcases hD : b.drop ro with _ | ⟨hd, r⟩
      · rw [hD] at hjd; simp at hjd
      · simp only [List.set_cons_zero, points_cons_append]
    · have := points_set_perm [(lo, m), (lo + m, k)] ((b.drop ro).set j (lo, m + k)) j lo (m + k)
        (List.getElem?_set_self hjd) (by simp [List.range'_append_1])
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.set_set] at this
      exact (this.foldl_eq' (fun x _ y _ z => hc z _ _) a).symm
  simp only [Op.leaf, htake]
  by_cases hcell : c ∈ points (b.take ro)
  · simp only [hcell, ↓reduceIte, hf2, hl1, hdrop, id, hred, hf, hla]
  · simp only [hcell, ↓reduceIte]

theorem atiles_part {ts : Nat} {mode : Mode} {x y : AAxis} (h : x.part = true ∨ x.dyn = true ∨ ts < x.cap)
    (hy : y ∈ atiles ts mode x) : y.part = true := by
  have hp : (x.part || x.dyn || decide (ts < x.cap)) = true := by
    rcases h with h | h | h <;> simp [h]
  simp only [atiles, hp] at hy
  repeat' split at hy
  all_goals simp at hy
  all_goals first | (rcases hy with rfl | rfl <;> rfl) | (subst hy; rfl) | skip

theorem atiles_red {ts : Nat} {mode : Mode} {x y : AAxis} (hy : y ∈ atiles ts mode x) : y.red = x.red := by
  simp only [atiles] at hy
  repeat' split at hy
  all_goals simp at hy
  all_goals first | (rcases hy with rfl | rfl <;> rfl) | (subst hy; rfl) | skip

/-- A loop that a `Split` cut stays cut below it. -/
theorem part_mono (e : Nat) : ∀ (t : Tree) (a : AState), (∃ x, a[e]? = some x ∧ x.part = true) →
    ∀ a' ∈ aleaves t a, ∃ x, a'[e]? = some x ∧ x.part = true
  | .leaf _, a, h, a', ha' => by
    simp only [aleaves, List.mem_singleton] at ha'
    exact ha' ▸ h
  | .split d ts mode body, a, h, a', ha' => by
    simp only [aleaves] at ha'
    split at ha'
    · rename_i x hx
      obtain ⟨y, hy, ha'⟩ := List.mem_flatMap.1 ha'
      refine part_mono e body (a.set d y) ?_ a' ha'
      obtain ⟨z, hz, hzp⟩ := h
      by_cases hde : d = e
      · subst hde
        rw [hx] at hz
        cases hz
        exact ⟨y, by simp [(List.getElem?_eq_some_iff.1 hx).1], atiles_part (.inl hzp) hy⟩
      · exact ⟨z, by rw [List.getElem?_set_ne hde]; exact hz, hzp⟩
    · simp at ha'
  | .reduce d w body, a, h, a', ha' => by
    simp only [aleaves] at ha'
    split at ha'
    · rename_i x hx
      refine part_mono e body _ ?_ a' ha'
      obtain ⟨z, hz, hzp⟩ := h
      by_cases hde : d = e
      · subst hde
        rw [hx] at hz
        cases hz
        exact ⟨{ x with red := w }, by simp [(List.getElem?_eq_some_iff.1 hx).1], hzp⟩
      · exact ⟨z, by rw [List.getElem?_set_ne hde]; exact hz, hzp⟩
    · simp at ha'
  | .pack _ _ _ _ body, a, h, a', ha' => part_mono e body a h a' ha'
  | .stage _ body, a, h, a', ha' => part_mono e body a h a' ha'
  | .alloc _ body, a, h, a', ha' => part_mono e body a h a' ha'
  | .switch d c small big, a, h, a', ha' => by
    simp only [aleaves] at ha'
    split at ha'
    · rename_i x hx
      have hs : ∃ z, (a.set d (asmall c x))[e]? = some z ∧ z.part = true := by
        obtain ⟨z, hz, hzp⟩ := h
        by_cases hde : d = e
        · subst hde
          rw [hx] at hz
          cases hz
          exact ⟨asmall c x, by simp [(List.getElem?_eq_some_iff.1 hx).1], hzp⟩
        · exact ⟨z, by rw [List.getElem?_set_ne hde]; exact hz, hzp⟩
      split at ha'
      · rcases List.mem_append.1 ha' with ha' | ha'
        · exact part_mono e small _ hs a' ha'
        · exact part_mono e big a h a' ha'
      · split at ha'
        · exact part_mono e small _ hs a' ha'
        · exact part_mono e big a h a' ha'
    · simp at ha'

theorem ordered_get {ro e : Nat} {a : AState} {x : AAxis} (h : ordered ro a = true) (he : ro < e)
    (hx : a[e]? = some x) : x.part = false := by
  simp only [ordered, List.all_eq_true, Bool.not_eq_true'] at h
  refine h x ?_
  obtain ⟨hlt, rfl⟩ := List.getElem?_eq_some_iff.1 hx
  have : e = ro + 1 + (e - (ro + 1)) := by omega
  rw [List.mem_iff_getElem]
  exact ⟨e - (ro + 1), by simp; omega, by simp only [List.getElem_drop]; congr 1; omega⟩

/-- A combiner without a merge only passes the checks where no leaf sees a later reduction loop cut. -/
theorem scheck_ordered {ro : Nat} : ∀ (t : Tree) (a : AState), scheck 0 ro t a = true →
    ∀ a' ∈ aleaves t a, ordered ro a' = true
  | .leaf _, a, h, a', ha' => by
    simp only [aleaves, List.mem_singleton] at ha'
    subst ha'
    simp only [scheck, leafOk, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at h
    exact h.1.1.resolve_left (by omega)
  | .split d ts mode body, a, h, a', ha' => by
    simp only [scheck] at h
    simp only [aleaves] at ha'
    split at h
    · rename_i x hx
      simp only [hx] at ha'
      simp only [Bool.and_eq_true, List.all_eq_true] at h
      obtain ⟨y, hy, ha'⟩ := List.mem_flatMap.1 ha'
      exact scheck_ordered body _ (h.2 y hy) a' ha'
    · simp at h
  | .reduce d w body, a, h, a', ha' => by
    simp only [scheck] at h
    simp only [aleaves] at ha'
    split at h
    · rename_i x hx
      simp only [hx] at ha'
      exact scheck_ordered body _ h a' ha'
    · simp at h
  | .pack _ _ _ _ body, a, h, a', ha' => by
    simp only [scheck, Bool.and_eq_true] at h
    exact scheck_ordered body a h.2 a' ha'
  | .stage _ body, a, h, a', ha' => scheck_ordered body a h a' ha'
  | .alloc _ body, a, h, a', ha' => scheck_ordered body a h a' ha'
  | .switch d c small big, a, h, a', ha' => by
    simp only [scheck] at h
    simp only [aleaves] at ha'
    split at h
    · rename_i x hx
      simp only [hx] at ha'
      split at h
      · rename_i hdyn
        simp only [hdyn, ite_true] at ha'
        simp only [Bool.and_eq_true] at h
        rcases List.mem_append.1 ha' with ha' | ha'
        · exact scheck_ordered small _ h.1 a' ha'
        · exact scheck_ordered big a h.2 a' ha'
      · rename_i hdyn
        simp only [hdyn] at ha'
        split at h
        · rename_i hlt
          simp only [hlt, ite_true] at ha'
          exact scheck_ordered small _ h a' ha'
        · rename_i hlt
          simp only [hlt, ite_false] at ha'
          exact scheck_ordered big a h a' ha'
    · simp at h

/-- A tree whose checks pass leaves in the output buffer what one `Leaf` on its box leaves there. -/
theorem Op.exec_eq_leaf (op : Op β α) {lanes ro : Nat} (rb : Box) :
    ∀ (t : Tree) (a : AState) (b : Box) (dst : Point → α), t.wf a.length ro = true → scheck lanes ro t a = true →
      Holds a b → Sub ro rb b → ((∀ a' ∈ aleaves t a, ordered ro a' = true) ∨ op.Comm) →
      op.exec ro rb t b dst = op.leaf ro rb b dst
  | .leaf _, a, b, dst, _, _, _, _, _ => by simp [Op.exec, run]
  | .pack _ _ _ _ body, a, b, dst, hw, hc, h, hs, ho => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    simp only [scheck, Bool.and_eq_true] at hc
    exact op.exec_eq_leaf rb body a b dst hw.2 hc.2 h hs ho
  | .stage _ body, a, b, dst, hw, hc, h, hs, ho => op.exec_eq_leaf rb body a b dst hw hc h hs ho
  | .alloc _ body, a, b, dst, hw, hc, h, hs, ho => op.exec_eq_leaf rb body a b dst hw hc h hs ho
  | .reduce d w body, a, b, dst, hw, hc, h, hs, ho => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    simp only [scheck] at hc
    simp only [aleaves] at ho
    split at hc
    · rename_i x hx
      simp only [hx] at ho
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hdb : d < b.length := h.1 ▸ hda
      have hxb : x.Holds b[d] := by
        have := h.2 d hda hdb
        rwa [(List.getElem?_eq_some_iff.1 hx).2] at this
      have hh : Holds (a.set d { x with red := w }) b := by
        have := h.set d (y := { x with red := w }) (tl := b[d]) hxb
        rwa [List.set_getElem_self hdb] at this
      exact op.exec_eq_leaf rb body _ b dst (by simpa using hw.2) hc hh hs ho
    · simp at hc
  | .switch d c small big, a, b, dst, hw, hc, h, hs, ho => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    simp only [scheck] at hc
    simp only [aleaves] at ho
    split at hc
    · rename_i x hx
      simp only [hx] at ho
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hdb : d < b.length := h.1 ▸ hda
      have hxb : x.Holds b[d] := by
        have := h.2 d hda hdb
        rwa [(List.getElem?_eq_some_iff.1 hx).2] at this
      have hself : b.set d b[d] = b := List.set_getElem_self hdb
      have hrun : op.exec ro rb (.switch d c small big) b dst =
          if b[d].2 < c then op.exec ro rb small b dst else op.exec ro rb big b dst := by
        simp only [Op.exec, run, List.getElem?_eq_getElem hdb]
        split <;> rfl
      rw [hrun]
      have hsmall : b[d].2 < c → Holds (a.set d (asmall c x)) b := fun hn => by
        have : (asmall c x).Holds b[d] := by
          refine ⟨by simp only [asmall]; have := hxb.1; omega, fun hd => ?_⟩
          have := hxb.2 hd
          simp only [asmall]
          omega
        simpa [hself] using h.set d this
      have hos : (∀ a' ∈ aleaves small (a.set d (asmall c x)) ++ aleaves big a, ordered ro a' = true) ∨ op.Comm →
          ((∀ a' ∈ aleaves small (a.set d (asmall c x)), ordered ro a' = true) ∨ op.Comm) ∧
          ((∀ a' ∈ aleaves big a, ordered ro a' = true) ∨ op.Comm) := fun ho =>
        ho.elim (fun ho => ⟨.inl fun a' ha' => ho a' (List.mem_append_left _ ha'),
          .inl fun a' ha' => ho a' (List.mem_append_right _ ha')⟩) fun hc => ⟨.inr hc, .inr hc⟩
      by_cases hdyn : x.dyn = true
      · simp only [hdyn, ite_true, Bool.and_eq_true] at hc ho
        obtain ⟨ho1, ho2⟩ := hos ho
        split
        · rename_i hn
          exact op.exec_eq_leaf rb small _ b dst (by simpa using hw.1.2) hc.1 (hsmall hn) hs ho1
        · exact op.exec_eq_leaf rb big a b dst hw.2 hc.2 h hs ho2
      · have hf : x.dyn = false := by simpa using hdyn
        have hn := hxb.2 hf
        simp only [hf, Bool.false_eq_true, ite_false] at hc ho
        rw [← hn] at hc ho
        split
        · rename_i hlt
          simp only [hlt, ite_true] at hc ho
          exact op.exec_eq_leaf rb small _ b dst (by simpa using hw.1.2) hc (hsmall hlt) hs ho
        · rename_i hlt
          simp only [hlt, ite_false] at hc ho
          exact op.exec_eq_leaf rb big a b dst hw.2 hc h hs ho
    · simp at hc
  | .split d ts mode body, a, b, dst, hw, hc, h, hs, ho => by
    simp only [Tree.wf, Bool.and_eq_true, decide_eq_true_eq] at hw
    obtain ⟨⟨_, hts⟩, hwb⟩ := hw
    simp only [scheck] at hc
    simp only [aleaves] at ho
    split at hc
    · rename_i x hx
      simp only [hx] at ho
      simp only [Bool.and_eq_true, List.all_eq_true] at hc
      have hall := hc.2
      have hda : d < a.length := (List.getElem?_eq_some_iff.1 hx).1
      have hdb : d < b.length := h.1 ▸ hda
      have hdr : d < rb.length := hs.1 ▸ hdb
      have hxb : x.Holds b[d] := by
        have := h.2 d hda hdb
        rwa [(List.getElem?_eq_some_iff.1 hx).2] at this
      have hself : b.set d b[d] = b := List.set_getElem_self hdb
      rcases hbd : b[d] with ⟨lo, n⟩
      rw [hbd] at hxb hself
      have hsub := fun hro => hs.2 d hro hdb hdr
      simp only [hbd] at hsub
      have hrun : op.exec ro rb (.split d ts mode body) b dst =
          (tiles ts lo n).foldl (fun dst tl => op.exec ro rb body (b.set d tl) dst) dst := by
        simp only [Op.exec, run, List.getElem?_eq_getElem hdb, hbd, List.foldl_flatMap]
      have htile : ∀ tl ∈ tiles ts lo n, ∀ dst, op.exec ro rb body (b.set d tl) dst = op.leaf ro rb (b.set d tl) dst := by
        intro tl htl dst
        obtain ⟨y, hy, hytl⟩ := atiles_sound (mode := mode) hxb htl
        have hr := mem_tiles_range htl
        refine op.exec_eq_leaf rb body (a.set d y) (b.set d tl) dst (by simpa using hwb) (hall y hy) (h.set d hytl)
          (hs.set d tl fun hro hr' => ?_) (ho.elim (fun ho => .inl fun a' ha' => ho a' (List.mem_flatMap.2 ⟨y, hy, ha'⟩)) .inr)
        have := hsub hro
        omega
      rw [hrun, foldl_congr_mem htile]
      by_cases hout : d < ro
      · by_cases hn : n = 0
        · subst hn
          have : tiles ts lo 0 = [] := by unfold tiles; simp
          rw [this, List.foldl_nil]
          refine (op.leaf_of_nil ro rb b dst (points_eq_nil_iff.2 ⟨(lo, 0), ?_, rfl⟩)).symm
          rw [List.mem_iff_getElem]
          exact ⟨d, by simp; omega, by simp [hbd]⟩
        · have := tiles_foldl (fun tl dst => op.leaf ro rb (b.set d tl) dst) hts lo n
            (fun l m k s _ _ _ _ => op.leaf_seq_out rb b l m k s hout hdb) n lo dst (Nat.le_refl _) (Nat.le_refl _) (by omega)
          rw [this, hself]
      · have hro : ro ≤ d := by omega
        have hsub := hsub hro
        obtain ⟨j, rfl⟩ : ∃ j, d = ro + j := ⟨d - ro, by omega⟩
        by_cases hcm : j = 0 ∨ op.Comm
        · have := tiles_foldl (fun tl dst => op.leaf ro rb (b.set (ro + j) tl) dst) hts lo n
            (fun l m k s h1 h2 h3 h4 => op.leaf_seq_red rb b j l m k s hdb hdr (by omega) (by omega) h3 h4 hcm)
            n lo dst (Nat.le_refl _) (Nat.le_refl _) hsub.2.2
          rw [this, hself]
        · have ho := ho.resolve_right fun hc => hcm (.inr hc)
          have hj : 0 < j := by omega
          by_cases hnts : n ≤ ts
          · rw [tiles_single hts hsub.2.2 hnts, List.foldl_cons, List.foldl_nil, hself]
          · obtain ⟨tl, htl⟩ : ∃ tl, tl ∈ tiles ts lo n := by
              unfold tiles
              rw [dite_eq_left ⟨hts, hsub.2.2⟩]
              exact ⟨_, List.mem_cons_self⟩
            obtain ⟨y, hy, hytl⟩ := atiles_sound (mode := mode) hxb htl
            have hyp : y.part = true := atiles_part (.inr (.inr (by have := hxb.1; simp only at this; omega))) hy
            have hempty : run body (b.set (ro + j) tl) = [] := by
              refine List.eq_nil_iff_forall_not_mem.2 fun lf hlf => ?_
              obtain ⟨a', ha', _⟩ := ghost_sound body _ _ (hall y hy) (h.set (ro + j) hytl) lf hlf
              obtain ⟨z, hz, hzp⟩ := part_mono (ro + j) body (a.set (ro + j) y) ⟨y, by simp [hda], hyp⟩ a' ha'
              have := ordered_get (ho a' (List.mem_flatMap.2 ⟨y, hy, ha'⟩)) (by omega) hz
              rw [this] at hzp
              cases hzp
            have hperm := run_perm (rn := b.length) body (b.set (ro + j) tl) (by simpa [h.1] using hwb) (by simp)
            rw [hempty] at hperm
            have hnil : points (b.set (ro + j) tl) = [] := by simpa using hperm.symm
            obtain ⟨e, he, he0⟩ := points_eq_nil_iff.1 hnil
            have htake : points (b.take ro) = [] := by
              rcases List.mem_or_eq_of_mem_set he with he | rfl
              · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem he
                by_cases hir : i < ro
                · refine points_eq_nil_iff.2 ⟨b[i], ?_, he0⟩
                  rw [List.mem_iff_getElem]
                  exact ⟨i, by simp; omega, by simp⟩
                · have := (hs.2 i (by omega) hi (hs.1 ▸ hi)).2.2
                  omega
              · have := (mem_tiles_range htl).2.2
                omega
            rw [op.leaf_of_nil ro rb b dst htake]
            have : ∀ tl ∈ tiles ts lo n, ∀ s : Point → α, op.leaf ro rb (b.set (ro + j) tl) s = s := fun tl _ s =>
              op.leaf_of_nil ro rb _ s (by rw [List.take_set_of_le (Nat.le_add_right ..)]; exact htake)
            rw [foldl_congr_mem this, foldl_id]
    · simp at hc

/-- From an uninitialized output buffer, a tree whose checks pass computes `post` of each cell's left fold. -/
theorem Op.exec_spec (op : Op β α) {lanes ro : Nat} (t : Tree) (a : AState) (b : Box) (dst : Point → α)
    (hc : check lanes ro t a = true) (h : Holds a b) (hpos : ∀ tl ∈ b.drop ro, 0 < tl.2)
    (hl : lanes = 0 ∨ op.Comm) :
    op.exec ro b t b dst =
      fun c => if c ∈ points (b.take ro) then op.post (op.red (b.drop ro) c op.init) else dst c := by
  simp only [check, Bool.and_eq_true] at hc
  have hs : Sub ro b b := ⟨rfl, fun d hd hb _ => ⟨Nat.le_refl _, Nat.le_refl _, hpos _ (by
    rw [List.mem_iff_getElem]
    exact ⟨d - ro, by simp; omega, by simp only [List.getElem_drop]; congr 1; omega⟩)⟩⟩
  have ho : (∀ a' ∈ aleaves t a, ordered ro a' = true) ∨ op.Comm := by
    rcases hl with rfl | hl
    · exact .inl (scheck_ordered t a hc.2)
    · exact .inr hl
  rw [op.exec_eq_leaf b t a b dst hc.1 hc.2 h hs ho]
  funext c
  simp [Op.leaf, fresh, last]

end Sched
