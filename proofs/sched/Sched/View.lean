import Sched.Aff
import Sched.Tree

/-!
# Views: what `Pack` and `Stage` do to an input

A view is how the loops below a node read one input slot: an origin `q`, the elements a step of `q` stands for, and a map `rd` from `q` and a tile-local point to the value read.
`Split` moves `q`, `Pack` replaces the view by one that reads a panel, and `Stage` replaces a producer by one that reads the box it materializes.
The theorem says that a leaf below any of them reads, at each of its points, the value its slot specifies there.
The checks only look at the ghost part of a view: whether it holds data, its unit, and the loops it reads.
-/

namespace Sched

abbrev Vec := Nat → Nat

def boxLo (b : Box) : Vec := fun d => (b.getD d (0, 1)).1
def boxSz (b : Box) : Vec := fun d => (b.getD d (0, 1)).2

/-- The static part of a view. -/
structure Ghost where
  ready : Bool
  unit  : Vec
  reads : Nat → Bool

structure View (Val : Type) where
  g    : Ghost
  q    : Vec
  /-- The box the view was built for: where it starts and its extents. -/
  home : Vec
  hsz  : Vec
  rd   : Vec → Vec → Val
  /-- The access map of a producer and how it fills the box `(blo, bext)`. -/
  vals : Nat → Nat
  rows : Nat → AffE
  fill : Vec → Vec → Vec → Val

variable {Val : Type}

def splitView (d t ts : Nat) (v : View Val) : View Val :=
  { v with q := fun e => if e = d then v.q d + t * (ts / v.g.unit d) else v.q e }

def packGhost (d w : Nat) (g : Ghost) : Ghost :=
  { g with ready := true, unit := fun e => if e = d then w else 1 }

/-- `levels_pack_one`: the panel holds the old view at `src` where it was filled, and anything elsewhere. -/
def packView (junk : Vec → Vec → Val) (rn d w : Nat) (dep : Nat → Bool) (lo sz : Vec) (v : View Val) : View Val :=
  { v with
    g := packGhost d w v.g
    q := fun _ => 0
    home := lo
    hsz := sz
    rd := fun q o =>
      let src : Vec := fun k =>
        if k = d then (q d + o d / w) * w + o d % w else if k < rn ∧ dep k = true then q k + o k else 0
      if src d < sz d ∧ ∀ k < rn, k ≠ d → dep k = true → src k < sz k then v.rd v.q src else junk q o }

def stageGhost (g : Ghost) : Ghost := { g with ready := true, unit := fun _ => 1 }

/-- `exec_stage`: the producer fills the box of what the tile at `lo` of extents `sz` reads, and the view reads that box. -/
def stageView (lo sz : Vec) (v : View Val) : View Val :=
  if v.g.ready then v else
    { v with
      g := stageGhost v.g
      q := fun _ => 0
      home := lo
      hsz := sz
      rd := fun q o =>
        let x : Vec := fun e => q e + o e
        v.fill (fun i => (affBox v.vals lo sz (v.rows i)).1) (fun i => (affBox v.vals lo sz (v.rows i)).2)
          fun i =>
            if ((v.rows i).span v.vals lo sz).1 then ((v.rows i).rel x).2
            else (v.rows i).eval v.vals (fun e => lo e + x e) - (affBox v.vals lo sz (v.rows i)).1 }

/-- The leaves of a run with the views each of them reads. -/
def runV (junk : Vec → Vec → Val) (rn : Nat) : Tree → List (View Val) → Box → List (Box × List (View Val) × Nat)
  | .leaf vdim, vs, b => [(b, vs, vdim)]
  | .split d ts _ body, vs, b =>
    match b[d]? with
    | some (lo, n) =>
      (tiles ts lo n).flatMap fun tl => runV junk rn body (vs.map (splitView d ((tl.1 - lo) / ts) ts)) (b.set d tl)
    | none => []
  | .reduce _ _ body, vs, b => runV junk rn body vs b
  | .pack d w slot dep body, vs, b =>
    match vs[slot]? with
    | some v => runV junk rn body (vs.set slot (packView junk rn d w (fun k => dep.getD k false) (boxLo b) (boxSz b) v)) b
    | none => []
  | .stage slot body, vs, b =>
    match vs[slot]? with
    | some v => runV junk rn body (vs.set slot (stageView (boxLo b) (boxSz b) v)) b
    | none => []
  | .alloc _ body, vs, b => runV junk rn body vs b
  | .switch d c small big, vs, b =>
    match b[d]? with
    | some (_, n) => if n < c then runV junk rn small vs b else runV junk rn big vs b
    | none => []

/-- The checks on views, evaluated on their ghost part alone. -/
def vcheck (rn : Nat) : Tree → List Ghost → Bool
  | .leaf _, gs => gs.all (·.ready)
  | .split d ts _ body, gs => gs.all (fun g => ts % g.unit d = 0) && vcheck rn body gs
  | .reduce _ _ body, gs => vcheck rn body gs
  | .pack d w slot dep body, gs =>
    match gs[slot]? with
    | some g =>
      g.ready && (List.range rn).all (fun k => dep.getD k false = g.reads k) &&
        vcheck rn body (gs.set slot (packGhost d w g))
    | none => false
  | .stage slot body, gs =>
    match gs[slot]? with
    | some g => vcheck rn body (gs.set slot (if g.ready then g else stageGhost g))
    | none => false
  | .alloc _ body, gs => vcheck rn body gs
  | .switch _ _ small big, gs => vcheck rn small gs && vcheck rn big gs

/-- The view supplies the values `spec` gives the points of the box `b`.
A view that holds data reads `spec` within its home box `(home, hsz)`, and `q` places `b` in it.
A producer has a result `res` that `spec` reads through its access map, and fills any box with it. -/
def View.Ok (rn : Nat) (spec : Vec → Val) (b : Box) (v : View Val) : Prop :=
  (∀ p p' : Vec, (∀ k < rn, v.g.reads k = true → p k = p' k) → spec p = spec p') ∧
  if v.g.ready then
    (∀ q o : Vec, (∀ e < rn, q e * v.g.unit e + o e < v.hsz e) → (∀ e, rn ≤ e → q e = 0 ∧ o e = 0) →
        v.rd q o = spec fun e => v.home e + q e * v.g.unit e + o e) ∧
      (∀ e < rn, (boxLo b) e = v.home e + v.q e * v.g.unit e ∧ (boxLo b) e + (boxSz b) e ≤ v.home e + v.hsz e) ∧
      (∀ e, rn ≤ e → v.q e = 0) ∧ ∀ e, 0 < v.g.unit e
  else
    ∃ res : Vec → Val,
      (∀ p, spec p = res fun i => (v.rows i).eval v.vals p) ∧ (∀ i, (v.rows i).WF) ∧
      ∀ blo bext y : Vec, (∀ i, y i < bext i) → v.fill blo bext y = res fun i => blo i + y i

def AllOk (rn : Nat) (specs : List (Vec → Val)) (b : Box) (vs : List (View Val)) : Prop :=
  vs.length = specs.length ∧ ∀ i (h : i < vs.length) (h' : i < specs.length), vs[i].Ok rn specs[i] b

theorem boxLo_set {b : Box} {d : Nat} (hd : d < b.length) (tl : Tile) (e : Nat) :
    boxLo (b.set d tl) e = if e = d then tl.1 else (boxLo b) e := by
  simp only [boxLo, List.getD_eq_getElem?_getD, List.getElem?_set]
  by_cases h : d = e
  · subst h; simp [hd]
  · have : ¬e = d := fun h' => h h'.symm
    simp [h, this]

theorem boxSz_set {b : Box} {d : Nat} (hd : d < b.length) (tl : Tile) (e : Nat) :
    boxSz (b.set d tl) e = if e = d then tl.2 else (boxSz b) e := by
  simp only [boxSz, List.getD_eq_getElem?_getD, List.getElem?_set]
  by_cases h : d = e
  · subst h; simp [hd]
  · have : ¬e = d := fun h' => h h'.symm
    simp [h, this]

theorem boxLo_ge {b : Box} {e : Nat} (h : b.length ≤ e) : (boxLo b) e = 0 := by
  simp [boxLo, List.getD_eq_getElem?_getD, List.getElem?_eq_none h]

theorem boxSz_ge {b : Box} {e : Nat} (h : b.length ≤ e) : (boxSz b) e = 1 := by
  simp [boxSz, List.getD_eq_getElem?_getD, List.getElem?_eq_none h]

theorem mem_tiles_start {ts lo n : Nat} {tl : Tile} (h : tl ∈ tiles ts lo n) : ∃ t, tl.1 = lo + t * ts := by
  induction n using Nat.strongRecOn generalizing lo with
  | _ n ih =>
    unfold tiles at h
    split at h
    · rename_i hn
      rcases List.mem_cons.1 h with rfl | h
      · exact ⟨0, by simp⟩
      · by_cases hle : n ≤ ts
        · have : n - min n ts = 0 := by omega
          rw [this] at h
          unfold tiles at h
          simp at h
        · obtain ⟨t, ht⟩ := ih (n - min n ts) (by omega) h
          have hmin : min n ts = ts := by omega
          exact ⟨t + 1, by rw [ht, hmin, Nat.add_mul]; omega⟩
    · simp at h

/-- A `Split` that cuts whole units keeps a view in its home box. -/
theorem View.Ok.split {rn : Nat} {spec : Vec → Val} {b : Box} {v : View Val} (h : v.Ok rn spec b)
    {d ts lo n : Nat} {tl : Tile} (hlen : b.length = rn) (hd : d < b.length) (hbd : b[d] = (lo, n)) (htl : tl ∈ tiles ts lo n)
    (hu : ts % v.g.unit d = 0) : (splitView d ((tl.1 - lo) / ts) ts v).Ok rn spec (b.set d tl) := by
  obtain ⟨hind, h⟩ := h
  refine ⟨hind, ?_⟩
  have hg : (splitView d ((tl.1 - lo) / ts) ts v).g = v.g := rfl
  rw [hg]
  by_cases hr : v.g.ready = true
  · simp only [hr, ite_true] at h ⊢
    obtain ⟨ha, hb, hq, hu'⟩ := h
    refine ⟨ha, fun e he => ?_, fun e he => ?_, hu'⟩
    · rw [boxLo_set hd, boxSz_set hd]
      by_cases hed : e = d
      · subst hed
        obtain ⟨t, ht⟩ := mem_tiles_start htl
        have hrange := mem_tiles_range htl
        have hts := (mem_tiles htl).1
        have hlo : (boxLo b) e = lo := by simp [boxLo, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hd, hbd]
        have hsz' : (boxSz b) e = n := by simp [boxSz, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hd, hbd]
        have hb := hb e he
        rw [hlo, hsz'] at hb
        have ht' : (tl.1 - lo) / ts = t := by
          rw [ht, Nat.add_sub_cancel_left, Nat.mul_div_cancel _ hts]
        have hdiv : ts / v.g.unit e * v.g.unit e = ts := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hu)
        simp only [splitView, ite_true, ht', Nat.add_mul, Nat.mul_assoc, hdiv]
        omega
      · simp only [splitView, hed, ite_false]
        exact hb e he
    · have hed : e ≠ d := by omega
      simp only [splitView, hed, ite_false]
      exact hq e he
  · simp only [hr] at h ⊢
    exact h

/-- A `Pack` whose `dep` is what its slot reads leaves the values of the slot as they were. -/
theorem View.Ok.pack {rn : Nat} {spec : Vec → Val} {b : Box} {v : View Val} (h : v.Ok rn spec b)
    (junk : Vec → Vec → Val) {d w : Nat} {dep : Nat → Bool} (hr : v.g.ready = true) (hd : d < rn) (hw : 0 < w)
    (hdep : ∀ k < rn, dep k = v.g.reads k) : (packView junk rn d w dep (boxLo b) (boxSz b) v).Ok rn spec b := by
  obtain ⟨hind, h⟩ := h
  simp only [hr, ite_true] at h
  obtain ⟨ha, hb, hq, _⟩ := h
  refine ⟨hind, ?_⟩
  have hready : (packView junk rn d w dep (boxLo b) (boxSz b) v).g.ready = true := rfl
  simp only [hready, ite_true]
  have hunit : ∀ e, (packView junk rn d w dep (boxLo b) (boxSz b) v).g.unit e = if e = d then w else 1 := fun _ => rfl
  refine ⟨fun q o hin hout => ?_, fun e _ => ⟨by simp [packView], Nat.le_refl _⟩, fun _ _ => rfl,
    fun e => by rw [hunit]; split <;> omega⟩
  have hhome : (packView junk rn d w dep (boxLo b) (boxSz b) v).home = boxLo b := rfl
  have hhsz : (packView junk rn d w dep (boxLo b) (boxSz b) v).hsz = boxSz b := rfl
  rw [hhome]
  rw [hhsz] at hin
  simp only [hunit] at hin ⊢
  have hsd : (q d + o d / w) * w + o d % w = q d * w + o d := by
    have := Nat.div_add_mod' (o d) w
    rw [Nat.add_mul]
    omega
  have hind' := hin d hd
  simp only [ite_true] at hind'
  have hfilled : (q d + o d / w) * w + o d % w < (boxSz b) d ∧
      ∀ k < rn, k ≠ d → dep k = true → (if k = d then (q d + o d / w) * w + o d % w
        else if k < rn ∧ dep k = true then q k + o k else 0) < (boxSz b) k := by
    refine ⟨by omega, fun k hk hkd hdk => ?_⟩
    have := hin k hk
    simp only [hkd, ite_false, Nat.mul_one] at this
    simp [hkd, hk, hdk, this]
  simp only [packView, ite_true]
  rw [ite_eq_left_of_eq_true _ _ (eq_true hfilled)]
  refine (ha v.q _ (fun e he => ?_) (fun e he => ?_)).trans (hind _ _ fun k hk hrk => ?_)
  · have hbe := hb e he
    have hine := hin e he
    by_cases hed : e = d
    · subst hed
      simp only [ite_true, hsd] at hine ⊢
      omega
    · simp only [hed, ite_false, Nat.mul_one] at hine ⊢
      split <;> omega
  · have hed : e ≠ d := by omega
    have : ¬(e < rn ∧ dep e = true) := fun h => by omega
    exact ⟨hq e he, by simp [hed, this]⟩
  · have hdk : dep k = true := (hdep k hk).trans hrk
    have hbk := (hb k hk).1
    by_cases hkd : k = d
    · subst hkd
      simp only [ite_true, hsd]
      omega
    · simp only [hkd, ite_false, hk, hdk, and_self, ite_true, Nat.mul_one]
      omega

/-- A `Stage` makes a producer a view that reads the producer's result where the slot's access map points. -/
theorem View.Ok.stage {rn : Nat} {spec : Vec → Val} {b : Box} {v : View Val} (h : v.Ok rn spec b)
    (hlen : b.length = rn) :
    (stageView (boxLo b) (boxSz b) v).Ok rn spec b ∧ (stageView (boxLo b) (boxSz b) v).g.ready = true := by
  by_cases hr : v.g.ready = true
  · have : stageView (boxLo b) (boxSz b) v = v := by simp [stageView, hr]
    rw [this]
    exact ⟨h, hr⟩
  · have hr' : v.g.ready = false := by simpa using hr
    obtain ⟨hind, h⟩ := h
    simp only [hr] at h
    obtain ⟨res, hspec, hwf, hfill⟩ := h
    simp only [stageView, hr', Bool.false_eq_true, ite_false]
    refine ⟨⟨hind, ?_⟩, rfl⟩
    simp only [stageGhost, ite_true]
    refine ⟨fun q o hin hout => ?_, fun e _ => ⟨by simp, Nat.le_refl _⟩, fun _ _ => trivial, fun _ => Nat.one_pos⟩
    have hx : ∀ e, q e + o e < (boxSz b) e := fun e => by
      by_cases he : e < rn
      · have := hin e he
        simp only [Nat.mul_one] at this
        exact this
      · have := hout e (by omega)
        rw [boxSz_ge (by omega), this.1, this.2]
        decide
    have hbox := fun i => affBox_sound v.vals (boxLo b) (boxSz b) (fun e => q e + o e) hx (v.rows i) (hwf i)
    rw [hfill _ _ _ fun i => ?_, hspec]
    · congr 1
      funext i
      obtain ⟨h1, _, h3⟩ := hbox i
      have he : (fun e => (boxLo b) e + q e * 1 + o e) = fun e => (boxLo b) e + (q e + o e) := by
        funext e
        omega
      rw [he]
      by_cases hl : ((v.rows i).span v.vals (boxLo b) (boxSz b)).1 = true
      · simp only [hl, ite_true]
        have := h3 hl
        omega
      · simp only [hl, Bool.false_eq_true, ite_false]
        omega
    · obtain ⟨h1, h2, h3⟩ := hbox i
      by_cases hl : ((v.rows i).span v.vals (boxLo b) (boxSz b)).1 = true
      · simp only [hl, ite_true]
        have := h3 hl
        omega
      · simp only [hl, Bool.false_eq_true, ite_false]
        omega

theorem AllOk.set {rn : Nat} {specs : List (Vec → Val)} {b : Box} {vs : List (View Val)} (h : AllOk rn specs b vs)
    {slot : Nat} {v : View Val} (hv : ∀ (_ : slot < vs.length) (h2 : slot < specs.length), v.Ok rn specs[slot] b) :
    AllOk rn specs b (vs.set slot v) := by
  refine ⟨by simpa using h.1, fun i hi hi' => ?_⟩
  simp only [List.getElem_set]
  split
  · rename_i he
    subst he
    exact hv (by simpa using hi) hi'
  · exact h.2 i (by simpa using hi) hi'

/-- Below any `Split`, `Pack` and `Stage` whose checks pass, a leaf's views hold data and supply the values their slots specify on its box. -/
theorem view_sound (junk : Vec → Vec → Val) {rn ro : Nat} (specs : List (Vec → Val)) :
    ∀ (t : Tree) (vs : List (View Val)) (b : Box), t.wf rn ro = true → vcheck rn t (vs.map (·.g)) = true →
      b.length = rn → AllOk rn specs b vs →
      ∀ lf ∈ runV junk rn t vs b, AllOk rn specs lf.1 lf.2.1 ∧ ∀ v ∈ lf.2.1, v.g.ready = true
  | .leaf _, vs, b, _, hc, _, h, lf, hl => by
    simp only [runV, List.mem_singleton] at hl
    subst hl
    simp only [vcheck, List.all_eq_true, List.mem_map] at hc
    exact ⟨h, fun v hv => hc v.g ⟨v, hv, rfl⟩⟩
  | .reduce _ _ body, vs, b, hw, hc, hb, h, lf, hl => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    exact view_sound junk specs body vs b hw.2 hc hb h lf hl
  | .alloc _ body, vs, b, hw, hc, hb, h, lf, hl => view_sound junk specs body vs b hw hc hb h lf hl
  | .switch d c small big, vs, b, hw, hc, hb, h, lf, hl => by
    simp only [Tree.wf, Bool.and_eq_true] at hw
    simp only [vcheck, Bool.and_eq_true] at hc
    simp only [runV] at hl
    split at hl
    · split at hl
      · exact view_sound junk specs small vs b hw.1.2 hc.1 hb h lf hl
      · exact view_sound junk specs big vs b hw.2 hc.2 hb h lf hl
    · simp at hl
  | .split d ts mode body, vs, b, hw, hc, hb, h, lf, hl => by
    simp only [Tree.wf, Bool.and_eq_true, decide_eq_true_eq] at hw
    simp only [vcheck, Bool.and_eq_true, List.all_eq_true, List.mem_map, decide_eq_true_eq] at hc
    simp only [runV] at hl
    have hd : d < b.length := hb ▸ hw.1.1
    rw [List.getElem?_eq_getElem hd] at hl
    rcases hbd : b[d] with ⟨lo, n⟩
    rw [hbd] at hl
    obtain ⟨tl, htl, hl⟩ := List.mem_flatMap.1 hl
    refine view_sound junk specs body _ (b.set d tl) hw.2 ?_ (by simpa using hb) ⟨by simpa using h.1, fun i hi hi' => ?_⟩ lf hl
    · have : (vs.map (splitView d ((tl.1 - lo) / ts) ts)).map (·.g) = vs.map (·.g) := by
        rw [List.map_map]
        rfl
      rw [this]
      exact hc.2
    · simp only [List.getElem_map]
      have hi0 : i < vs.length := by simpa using hi
      exact (h.2 i hi0 hi').split hb hd hbd htl (hc.1 vs[i].g ⟨vs[i], List.getElem_mem hi0, rfl⟩)
  | .pack d w slot dep body, vs, b, hw, hc, hb, h, lf, hl => by
    simp only [Tree.wf, Bool.and_eq_true, decide_eq_true_eq] at hw
    simp only [vcheck] at hc
    simp only [runV] at hl
    split at hl
    · rename_i v hv
      obtain ⟨hs, rfl⟩ := List.getElem?_eq_some_iff.1 hv
      simp only [List.getElem?_map, hv, Option.map_some, Bool.and_eq_true, List.all_eq_true, List.mem_range,
        decide_eq_true_eq] at hc
      refine view_sound junk specs body _ b hw.2 ?_ hb (h.set fun h1 h2 => ?_) lf hl
      · rw [List.map_set]
        exact hc.2
      · exact (h.2 slot h1 h2).pack junk hc.1.1 hw.1.1 hw.1.2 hc.1.2
    · simp at hl
  | .stage slot body, vs, b, hw, hc, hb, h, lf, hl => by
    simp only [vcheck] at hc
    simp only [runV] at hl
    split at hl
    · rename_i v hv
      obtain ⟨hs, rfl⟩ := List.getElem?_eq_some_iff.1 hv
      simp only [List.getElem?_map, hv, Option.map_some] at hc
      refine view_sound junk specs body _ b hw ?_ hb (h.set fun h1 h2 => ((h.2 slot h1 h2).stage hb).1) lf hl
      rw [List.map_set]
      have : (stageView (boxLo b) (boxSz b) vs[slot]).g = if vs[slot].g.ready then vs[slot].g else stageGhost vs[slot].g := by
        simp only [stageView]
        split <;> rfl
      rw [this]
      exact hc
    · simp at hl

/-- A point as the vector of its coordinates. -/
def vecOf (p : Point) : Vec := fun e => p.getD e 0

theorem mem_points_bounds : ∀ {b : Box} {p : Point}, p ∈ points b →
    p.length = b.length ∧ ∀ e, (boxLo b) e ≤ vecOf p e ∧ (e < b.length → vecOf p e < (boxLo b) e + (boxSz b) e)
  | [], p, h => by
    simp only [points, List.mem_singleton] at h
    subst h
    simp [boxLo, vecOf]
  | (lo, n) :: r, p, h => by
    obtain ⟨i, p', rfl, h1, h2, hp⟩ := mem_points_cons.1 h
    obtain ⟨hlen, ih⟩ := mem_points_bounds hp
    refine ⟨by simp [hlen], fun e => ?_⟩
    cases e with
    | zero => simp [boxLo, boxSz, vecOf, h1, h2]
    | succ e =>
      have := ih e
      simpa [boxLo, boxSz, vecOf] using this

/-- A leaf reads the specified value at each of its points. -/
theorem View.Ok.read {rn : Nat} {spec : Vec → Val} {b : Box} {v : View Val} (h : v.Ok rn spec b)
    (hr : v.g.ready = true) (hb : b.length = rn) {p : Point} (hp : p ∈ points b) :
    v.rd v.q (fun e => vecOf p e - (boxLo b) e) = spec (vecOf p) := by
  obtain ⟨hind, h⟩ := h
  simp only [hr, ite_true] at h
  obtain ⟨ha, hbq, hq, _⟩ := h
  obtain ⟨hlen, hbd⟩ := mem_points_bounds hp
  refine (ha v.q _ (fun e he => ?_) (fun e he => ?_)).trans (hind _ _ fun k hk _ => ?_)
  · have := hbq e he
    have := hbd e
    have := this.2 (by omega)
    omega
  · refine ⟨hq e he, ?_⟩
    have : vecOf p e = 0 := by simp [vecOf, List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega : p.length ≤ e)]
    omega
  · have := hbq k hk
    have := hbd k
    omega

/-- A slot that holds its data from the start reads it at the global point. -/
def plainView (spec : Vec → Val) (reads : Nat → Bool) (b : Box) : View Val :=
  { g := ⟨true, fun _ => 1, reads⟩, q := (boxLo b), home := fun _ => 0, hsz := fun e => boxLo b e + boxSz b e,
    rd := fun q o => spec fun e => q e + o e,
    vals := fun _ => 0, rows := fun _ => .lit 0, fill := fun _ _ _ => spec fun _ => 0 }

theorem plainView_ok {rn : Nat} (spec : Vec → Val) (reads : Nat → Bool) (b : Box) (hb : b.length = rn)
    (hind : ∀ p p' : Vec, (∀ k < rn, reads k = true → p k = p' k) → spec p = spec p') :
    (plainView spec reads b).Ok rn spec b := by
  refine ⟨hind, ?_⟩
  simp only [plainView, ite_true]
  refine ⟨fun q o _ _ => ?_, fun e _ => by simp, fun e he => boxLo_ge (by omega), fun _ => Nat.one_pos⟩
  congr 1
  funext e
  simp

theorem AffE.eval_congr (vals : Nat → Nat) {p p' : Vec} : ∀ e : AffE, (∀ k, e.mentions k = true → p k = p' k) →
    e.eval vals p = e.eval vals p' := by
  intro e
  induction e with
  | dim d => intro h; exact h d (by simp [AffE.mentions])
  | sym _ | lit _ => intro _; rfl
  | add a b iha ihb =>
    intro h
    simp only [AffE.eval]
    rw [iha fun k hk => h k (by simp [AffE.mentions, hk]), ihb fun k hk => h k (by simp [AffE.mentions, hk])]
  | mul a c ih | fdiv a c ih | cdiv a c ih | mod a c ih =>
    intro h
    simp only [AffE.eval]
    rw [ih fun k hk => h k (by simpa [AffE.mentions] using hk)]

end Sched
