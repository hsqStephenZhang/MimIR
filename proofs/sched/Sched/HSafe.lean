import Sched.HMachine
import Sched.Alloc

/-!
# Memory safety of the machine on the heap

A tree whose static checks pass runs without a fault on the heap, whatever the run-time extents are:
every read is in bounds of a live block that holds the fill the reader expects, every fill fits its block, and every block is freed.
Afterwards the heap is as before, up to what the storage of an `Alloc` above holds.
-/

namespace Sched

/-- What the run keeps true of the state and the heap. -/
structure Inv (rn : Nat) (g : HG) (dv : HD) (b : Box) (H : Heap) : Prop where
  holds : Holds g.a b
  len   : b.length = rn
  pos   : ∀ e < rn, 0 < boxSz b e
  unit  : ∀ e, 0 < g.unit e
  /-- `q` places the box in the view's home box. -/
  place : g.ready = true → ∀ e < rn,
    boxLo b e = dv.home e + dv.q e * g.unit e ∧ boxLo b e + boxSz b e ≤ dv.home e + dv.hsz e
  /-- The view's block is live, holds the fill the view reads, and has room for every panel of the home box. -/
  blk   : g.ready = true → ∃ blk, H dv.loc = some blk ∧ blk.tag = some dv.tag ∧
    ∀ q o : Vec, (∀ e < rn, g.reads e = true → (q e * g.unit e + o e) / g.unit e < AffE.up (g.unit e) (dv.hsz e)) →
      inShape (dv.idx q o) blk.shape = true
  loc   : g.ready = true → dv.loc ≤ g.depth ∧ ∀ sid cap, g.store = some (sid, cap) → sid ≠ dv.loc
  /-- Nothing is allocated beyond the depth. -/
  top   : ∀ i, g.depth < i → H i = none
  store : ∀ sid cap, g.store = some (sid, cap) → sid ≤ g.depth ∧ ∃ blk, H sid = some blk ∧ blk.cap = cap

/-- A run leaves the heap as it was, except for what the storage on offer holds. -/
def Frame (g : HG) (H H' : Heap) : Prop :=
  (∀ i, (∀ cap, g.store ≠ some (i, cap)) → H' i = H i) ∧
    ∀ sid cap, g.store = some (sid, cap) → ∃ blk, H' sid = some blk ∧ blk.cap = cap

/-- The rows of the slot's access map are well-formed and only mention loops the slot reads. -/
def Rows (rn : Nat) (g : HG) : Prop :=
  ∀ r ∈ g.rows, r.WF ∧ ∀ k, r.mentions k = true → k < rn ∧ g.reads k = true

theorem box_range (rn : Nat) (f : Nat → Nat) {e : Nat} (he : e < rn) :
    boxLo ((List.range rn).map fun k => (0, f k)) e = 0 ∧ boxSz ((List.range rn).map fun k => (0, f k)) e = f e := by
  simp [boxLo, boxSz, List.getD_eq_getElem?_getD, he]

theorem mem_points_range {rn : Nat} {f : Nat → Nat} {o : Point} (h : o ∈ points ((List.range rn).map fun k => (0, f k))) :
    ∀ e < rn, vecOf o e < f e := by
  intro e he
  have := (mem_points_bounds h).2 e
  rw [(box_range rn f he).1, (box_range rn f he).2] at this
  have := this.2 (by simpa using he)
  omega

theorem up_one (n : Nat) : AffE.up 1 n = n := by simp [AffE.up]

/-- The tile's ghost state is one of those the checks saw, and it holds of the tile. -/
theorem atile_sound {ts lo n : Nat} {mode : Mode} {x : AAxis} {tl : Tile} (hx : x.Holds (lo, n))
    (htl : tl ∈ tiles ts lo n) : atile ts mode x tl ∈ atiles ts mode x ∧ (atile ts mode x tl).Holds tl := by
  obtain ⟨hts, hsz⟩ := mem_tiles htl
  obtain ⟨hcap, hst⟩ := hx
  simp only at hcap hst
  have hmod := Nat.mod_lt n hts
  have hmle := Nat.mod_le n ts
  have hbound : tl.2 ≤ min x.cap ts := by rcases hsz with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
  simp only [atile, atiles]
  generalize (x.part || x.dyn || decide (ts < x.cap)) = part
  by_cases hdyn : x.dyn = true
  · simp only [hdyn, ite_true]
    by_cases hp : mode = .peel
    · simp only [hp, true_and, ite_true]
      by_cases hw : tl.2 = ts
      · simp only [hw, ite_true]
        exact ⟨by simp, by simp [AAxis.Holds, hw]⟩
      · simp only [hw, ite_false]
        exact ⟨by simp, hbound, by simp⟩
    · simp only [hp, false_and, ite_false]
      exact ⟨by simp, hbound, by simp⟩
  · have hf : x.dyn = false := by simpa using hdyn
    have hn := hst hf
    simp only [hf, Bool.false_eq_true, ite_false]
    rw [← hn]
    by_cases hl : mode = .loop ∧ n / ts ≠ 0 ∧ n % ts ≠ 0
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hl), ite_eq_left_of_eq_true _ _ (eq_true hl)]
      exact ⟨by simp, hn ▸ hbound, by simp⟩
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hl), ite_eq_right_of_eq_false _ _ (eq_false hl)]
      refine ⟨?_, Nat.le_refl _, fun _ => rfl⟩
      rcases hsz with ⟨h, hle⟩ | ⟨h, hne⟩
      · have : n / ts ≠ 0 := Nat.ne_of_gt (Nat.div_pos hle hts)
        exact List.mem_append_left _ (by rw [ite_eq_left_of_eq_true _ _ (eq_true this), h]; simp)
      · exact List.mem_append_right _ (by rw [ite_eq_left_of_eq_true _ _ (eq_true hne), h]; simp)

/-- A read at most `r e` into a loop lies in a panel of the home box, if `r` is the extent of the box or, along a loop packed at least that wide, anything up to the panel's width. -/
theorem read_panel {rn : Nat} {g : HG} {dv : HD} {b : Box} {H : Heap} (inv : Inv rn g dv b H) (hr : g.ready = true)
    (o : Vec) {e : Nat} (he : e < rn) (ho : o e < boxSz b e ∨ o e < g.unit e) :
    (dv.q e * g.unit e + o e) / g.unit e < AffE.up (g.unit e) (dv.hsz e) := by
  have hbe := inv.place hr e he
  have hue := inv.unit e
  have hpe := inv.pos e he
  rcases ho with ho | ho
  · exact div_lt_up hue (by omega)
  · have hq : (dv.q e * g.unit e + o e) / g.unit e = dv.q e := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hue, Nat.div_eq_of_lt ho, Nat.zero_add]
    rw [hq]
    have hc := AffE.le_mul_up hue (dv.hsz e)
    refine Nat.lt_of_mul_lt_mul_left (a := g.unit e) ?_
    rw [Nat.mul_comm]
    omega

theorem capOf_get {a : AState} {e : Nat} (h : e < a.length) : capOf a e = a[e].cap := by
  simp [capOf, List.getElem?_eq_getElem h]

theorem boxSz_get {b : Box} {e : Nat} (h : e < b.length) : boxSz b e = b[e].2 := by
  simp [boxSz, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

theorem boxLo_get {b : Box} {e : Nat} (h : e < b.length) : boxLo b e = b[e].1 := by
  simp [boxLo, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- The extents of a box stay within the `cap`s of its ghost state. -/
theorem Holds.sz_le {a : AState} {b : Box} (h : Holds a b) {e : Nat} (he : e < b.length) : boxSz b e ≤ capOf a e := by
  have hea : e < a.length := h.1 ▸ he
  rw [boxSz_get he, capOf_get hea]
  exact (h.2 e hea he).1

theorem Inv.ownPack {rn : Nat} {g : HG} {dv : HD} {b : Box} {H : Heap} (inv : Inv rn g dv b H) (s : Nat) :
    Inv rn (ownPack s g) dv b H :=
  ⟨inv.holds, inv.len, inv.pos, inv.unit, inv.place, inv.blk, inv.loc, inv.top, inv.store⟩

theorem Inv.ownStage {rn : Nat} {g : HG} {dv : HD} {b : Box} {H : Heap} (inv : Inv rn g dv b H) (s : Nat) :
    Inv rn (ownStage s g) dv b H :=
  ⟨inv.holds, inv.len, inv.pos, inv.unit, inv.place, inv.blk, inv.loc, inv.top, inv.store⟩

theorem Frame.refl {rn : Nat} {g : HG} {dv : HD} {b : Box} {H : Heap} (inv : Inv rn g dv b H) : Frame g H H :=
  ⟨fun _ _ => rfl, fun sid cap h => (inv.store sid cap h).2⟩

theorem Frame.trans {g : HG} {H H₁ H₂ : Heap} (h₁ : Frame g H H₁) (h₂ : Frame g H₁ H₂) : Frame g H H₂ :=
  ⟨fun i hi => (h₂.1 i hi).trans (h₁.1 i hi), h₂.2⟩

/-- The invariant only looks at the heap where a run that respects the frame leaves it alone. -/
theorem Inv.frame {rn : Nat} {g : HG} {dv : HD} {b : Box} {H H' : Heap} (inv : Inv rn g dv b H) (hf : Frame g H H') :
    Inv rn g dv b H' where
  holds := inv.holds
  len := inv.len
  pos := inv.pos
  unit := inv.unit
  place := inv.place
  blk := fun hr => by
    obtain ⟨blk, hb, ht, hs⟩ := inv.blk hr
    refine ⟨blk, ?_, ht, hs⟩
    rw [hf.1 dv.loc fun cap hc => (inv.loc hr).2 dv.loc cap hc rfl]
    exact hb
  loc := inv.loc
  top := fun i hi => by
    rw [hf.1 i fun cap hc => absurd (inv.store i cap hc).1 (Nat.not_le.2 hi)]
    exact inv.top i hi
  store := fun sid cap h => ⟨(inv.store sid cap h).1, hf.2 sid cap h⟩

theorem Inv.split {rn : Nat} {g : HG} {dv : HD} {b : Box} {H : Heap} (inv : Inv rn g dv b H) {d ts lo n : Nat}
    {mode : Mode} {x : AAxis} {tl : Tile} (hd : d < b.length) (hbd : b[d] = (lo, n)) (hx : g.a[d]? = some x)
    (htl : tl ∈ tiles ts lo n) (hu : ts % g.unit d = 0) :
    Inv rn { g with a := g.a.set d (atile ts mode x tl) }
      { dv with q := fun e => if e = d then dv.q d + (tl.1 - lo) / ts * (ts / g.unit d) else dv.q e } (b.set d tl) H where
  holds := by
    obtain ⟨hda, hxe⟩ := List.getElem?_eq_some_iff.1 hx
    have hxb : x.Holds (lo, n) := by
      have := inv.holds.2 d hda hd
      rwa [hxe, hbd] at this
    exact inv.holds.set d (atile_sound hxb htl).2
  len := by simpa using inv.len
  pos := fun e he => by
    rw [boxSz_set hd]
    split
    · exact (mem_tiles_range htl).2.2
    · exact inv.pos e he
  unit := inv.unit
  place := fun hr e he => by
    have hb := inv.place hr e he
    rw [boxLo_set hd, boxSz_set hd]
    by_cases hed : e = d
    · subst hed
      obtain ⟨t, ht⟩ := mem_tiles_start htl
      have hrange := mem_tiles_range htl
      have hts := (mem_tiles htl).1
      rw [boxLo_get hd, boxSz_get hd, hbd] at hb
      simp only at hb
      have ht' : (tl.1 - lo) / ts = t := by rw [ht, Nat.add_sub_cancel_left, Nat.mul_div_cancel _ hts]
      have hdiv : ts / g.unit e * g.unit e = ts := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hu)
      simp only [ite_true, ht', Nat.add_mul, Nat.mul_assoc, hdiv]
      omega
    · simp only [hed, ite_false]
      exact hb
  blk := inv.blk
  loc := inv.loc
  top := inv.top
  store := inv.store

/-- The panels of a `Pack` have room for every panel of its box. -/
theorem pack_inShape {rn : Nat} {a : AState} {b : Box} (hh : Holds a b) (hlen : b.length = rn) {d w : Nat}
    {dep reads : Nat → Bool} (hw : 0 < w) (hd : d < rn) (hrd : reads d = true) (hdep : ∀ k < rn, dep k = reads k)
    (q o : Vec)
    (h : ∀ e < rn, reads e = true →
      (q e * (if e = d then w else 1) + o e) / (if e = d then w else 1) < AffE.up (if e = d then w else 1) (boxSz b e)) :
    inShape (packIdx rn d w dep q o) (packShape rn d w dep a) = true := by
  refine inShape_append (inShape_map _ _ _ fun k hk => ?_) (by simp [inShape, Nat.mod_lt _ hw])
  have hk : k < rn := List.mem_range.1 hk
  have hsz := hh.sz_le (e := k) (hlen ▸ hk)
  by_cases hkd : k = d
  · subst hkd
    have := h k hk hrd
    simp only [ite_true] at this ⊢
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hw, Nat.add_comm] at this
    have := AffE.up_mono w hsz
    omega
  · simp only [hkd, ite_false]
    by_cases hdk : dep k = true
    · have := h k hk ((hdep k hk).symm.trans hdk)
      simp only [hkd, ite_false, Nat.mul_one, Nat.div_one, up_one] at this
      simp only [hdk, ite_true]
      omega
    · simp [hdk]

/-- The box of a `Stage` has room for what its tile reads. -/
theorem stage_inShape {rn : Nat} {g : HG} (hrows : Rows rn g) (b : Box) (q o : Vec)
    (h : ∀ e < rn, g.reads e = true → (q e * 1 + o e) / 1 < AffE.up 1 (boxSz b e)) :
    inShape (stageIdx g b q o) (stageShape g b) = true := by
  refine inShape_map _ _ _ fun r hr => ?_
  obtain ⟨hwf, hm⟩ := hrows r hr
  have hs := AffE.span_sound_of_mentions g.vals (boxLo b) (boxSz b) (fun e => q e + o e) r hwf fun k hk => by
    have := h k (hm k hk).1 (hm k hk).2
    simpa only [Nat.mul_one, Nat.div_one, up_one] using this
  simp only [affBox]
  by_cases hl : (r.span g.vals (boxLo b) (boxSz b)).1 = true
  · have := AffE.rel_sound g.vals (boxLo b) (boxSz b) (fun e => q e + o e) r hl
    simp only [hl, ite_true]
    omega
  · simp only [hl, Bool.false_eq_true, ite_false]
    exact Nat.lt_succ_of_le (Nat.sub_le_of_le_add (by rw [Nat.add_comm]; exact hs.2))

/-- A node that allocates its own block: it allocates, reads what it needs, fills, runs its body, and frees; the heap is then as before. -/
theorem own_block {H : Heap} {id : Addr} (hid : H id = none) (c : Nat) (shape : List Nat) (b : Box)
    (pre tr : List HEv) (hc : prod shape ≤ c)
    (hpre : hreplay (hset H id (some ⟨c, [], none⟩)) pre = some (hset H id (some ⟨c, [], none⟩)))
    (hbody : ∃ H3, hreplay (hset H id (some ⟨c, shape, some b⟩)) tr = some H3 ∧
      ∀ i, H3 i = hset H id (some ⟨c, shape, some b⟩) i) :
    ∃ H', hreplay H ([.alloc id c] ++ (pre ++ ([.fill id shape b] ++ (tr ++ [.free id])))) = some H' ∧ ∀ i, H' i = H i := by
  obtain ⟨H3, e3, f3⟩ := hbody
  have h3id : H3 id = some ⟨c, shape, some b⟩ := by rw [f3, hset_same]
  refine ⟨hset H3 id none, ?_, fun i => ?_⟩
  · simp only [List.singleton_append, hreplay, hstep, hid, Option.bind_some, hreplay_append, hpre, hset_same, hc,
      ite_true, hset_hset, e3, h3id]
  · by_cases h : i = id
    · rw [h, hset_same, hid]
    · rw [hset_ne _ _ h, f3, hset_ne _ _ h]

/-- A node that takes the storage of an `Alloc`: it reads what it needs, fills the storage, and runs its body. -/
theorem store_block {H : Heap} {sid : Addr} {blk0 : Blk} (hs : H sid = some blk0) (shape : List Nat) (b : Box)
    (pre tr : List HEv) (hc : prod shape ≤ blk0.cap) (hpre : hreplay H pre = some H)
    (hbody : ∃ H3, hreplay (hset H sid (some ⟨blk0.cap, shape, some b⟩)) tr = some H3 ∧
      ∀ i, H3 i = hset H sid (some ⟨blk0.cap, shape, some b⟩) i) :
    ∃ H', hreplay H (pre ++ ([.fill sid shape b] ++ tr)) = some H' ∧ (∀ i, i ≠ sid → H' i = H i) ∧
      ∃ blk, H' sid = some blk ∧ blk.cap = blk0.cap := by
  obtain ⟨H3, e3, f3⟩ := hbody
  refine ⟨H3, ?_, fun i hi => ?_, ⟨blk0.cap, shape, some b⟩, ?_, rfl⟩
  · simp only [hreplay_append, hpre, Option.bind_some, List.singleton_append, hreplay, hstep, hs, hc, ite_true, e3]
  · rw [f3, hset_ne _ _ hi]
  · rw [f3, hset_same]

/-- The state below a `Pack` or `Stage`: the view reads the node's block from the corner of its box. -/
theorem Inv.fresh {rn : Nat} {g : HG} {dv : HD} {b : Box} {H : Heap} (inv : Inv rn g dv b H) (g2 : HG) (dv2 : HD)
    (H2 : Heap) (ha : g2.a = g.a) (hu : ∀ e, 0 < g2.unit e) (hst : g2.store = none) (hq : ∀ e, dv2.q e = 0)
    (hhome : dv2.home = boxLo b) (hhsz : dv2.hsz = boxSz b)
    (hblk : ∃ blk, H2 dv2.loc = some blk ∧ blk.tag = some dv2.tag ∧
      ∀ q o : Vec, (∀ e < rn, g2.reads e = true → (q e * g2.unit e + o e) / g2.unit e < AffE.up (g2.unit e) (dv2.hsz e)) →
        inShape (dv2.idx q o) blk.shape = true)
    (hloc : dv2.loc ≤ g2.depth) (htop : ∀ i, g2.depth < i → H2 i = none) : Inv rn g2 dv2 b H2 where
  holds := ha ▸ inv.holds
  len := inv.len
  pos := inv.pos
  unit := hu
  place := fun _ e _ => by simp [hq, hhome, hhsz]
  blk := fun _ => hblk
  loc := fun _ => ⟨hloc, fun sid cap h => by rw [hst] at h; cases h⟩
  top := htop
  store := fun sid cap h => by rw [hst] at h; cases h

/-- The main theorem: a tree whose checks pass runs on the heap without a fault, and leaves the heap as it was, up to what the storage on offer holds. -/
theorem htrace_safe {rn ro slot : Nat} : ∀ (t : Tree) (g : HG) (dv : HD) (b : Box) (H : Heap),
    hcheck rn ro slot t g = true → Rows rn g → Inv rn g dv b H →
    ∃ H', hreplay H (htrace rn ro slot t g dv b) = some H' ∧ Frame g H H'
  | .leaf vdim, g, dv, b, H, hc, _, inv => by
    simp only [hcheck, Bool.and_eq_true] at hc
    obtain ⟨hr, hedge⟩ := hc
    obtain ⟨blk, hb, ht, hs⟩ := inv.blk hr
    simp only [htrace, hr, ite_true]
    refine ⟨H, hreplay_reads H dv.loc dv.tag blk hb ht (fun o => dv.idx dv.q (vecOf o)) _ fun o ho => ?_, Frame.refl inv⟩
    refine hs dv.q (vecOf o) fun e he hre => read_panel inv hr (vecOf o) he ?_
    have hoe := mem_points_range ho e he
    by_cases hcase : e < ro ∧ full g vdim = true
    · simp only [hcase, and_self, ite_true] at hoe
      have hel : e < b.length := inv.len ▸ he
      have hea : e < g.a.length := inv.holds.1 ▸ hel
      rw [capOf_get hea] at hoe
      have hx := inv.holds.2 e hea hel
      by_cases hdyn : g.a[e].dyn = true
      · simp only [hcase.2, Bool.not_true, Bool.false_or, List.all_eq_true, List.mem_range] at hedge
        have := hedge e hcase.1
        simp only [List.getElem?_eq_getElem hea, hdyn, hre, Bool.not_true, Bool.false_or, decide_eq_true_eq] at this
        exact .inr (by omega)
      · have := hx.2 (by simpa using hdyn)
        rw [boxSz_get hel]
        exact .inl (by omega)
    · simp only [hcase, ite_false] at hoe
      exact .inl hoe
  | .reduce _ _ body, g, dv, b, H, hc, hrows, inv => by
    simp only [hcheck] at hc
    simp only [htrace]
    exact htrace_safe body g dv b H hc hrows inv
  | .split d ts mode body, g, dv, b, H, hc, hrows, inv => by
    simp only [hcheck] at hc
    cases hx : g.a[d]? with
    | none => rw [hx] at hc; simp at hc
    | some x =>
      rw [hx] at hc
      simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hc
      obtain ⟨⟨_, hu⟩, hall⟩ := hc
      obtain ⟨hda, hxe⟩ := List.getElem?_eq_some_iff.1 hx
      have hdb : d < b.length := inv.holds.1 ▸ hda
      rcases hbd : b[d] with ⟨lo, n⟩
      have hbd' : b[d]? = some (lo, n) := by rw [List.getElem?_eq_getElem hdb, hbd]
      have hxb : x.Holds (lo, n) := by
        have := inv.holds.2 d hda hdb
        rwa [hxe, hbd] at this
      simp only [htrace, hbd', hx]
      exact hreplay_flatMap _ (Frame g H) _ H (Frame.refl inv) fun tl htl h hf => by
        have inv' := (inv.frame hf).split (mode := mode) hdb hbd hx htl hu
        obtain ⟨h2, e2, f2⟩ := htrace_safe body _ _ (b.set d tl) h (hall _ (atile_sound (mode := mode) hxb htl).1) hrows inv'
        exact ⟨h2, e2, hf.trans f2⟩
  | .switch d c small big, g, dv, b, H, hc, hrows, inv => by
    simp only [hcheck] at hc
    cases hx : g.a[d]? with
    | none => rw [hx] at hc; simp at hc
    | some x =>
      rw [hx] at hc
      obtain ⟨hda, hxe⟩ := List.getElem?_eq_some_iff.1 hx
      have hdb : d < b.length := inv.holds.1 ▸ hda
      rcases hbd : b[d] with ⟨lo, n⟩
      have hbd' : b[d]? = some (lo, n) := by rw [List.getElem?_eq_getElem hdb, hbd]
      have hxb : x.Holds (lo, n) := by
        have := inv.holds.2 d hda hdb
        rwa [hxe, hbd] at this
      simp only [htrace, hbd', hx]
      have hsmall : n < c → Inv rn { g with a := g.a.set d (asmall c x) } dv b H := fun hn => by
        have hh : Holds (g.a.set d (asmall c x)) b := by
          have : (asmall c x).Holds (lo, n) := by
            refine ⟨by simp only [asmall]; have := hxb.1; simp only at this; omega, fun hd => ?_⟩
            have := hxb.2 hd
            simp only [asmall] at this ⊢
            omega
          have := inv.holds.set d this
          rwa [← hbd, List.set_getElem_self] at this
        exact ⟨hh, inv.len, inv.pos, inv.unit, inv.place, inv.blk, inv.loc, inv.top, inv.store⟩
      by_cases hdyn : x.dyn = true
      · simp only [hdyn, ite_true, Bool.and_eq_true] at hc
        split
        · rename_i hn
          exact htrace_safe small _ dv b H hc.1 hrows (hsmall hn)
        · exact htrace_safe big g dv b H hc.2 hrows inv
      · have hf : x.dyn = false := by simpa using hdyn
        have hn := hxb.2 hf
        simp only at hn
        simp only [hf, Bool.false_eq_true, ite_false] at hc
        rw [← hn] at hc
        split
        · rename_i hlt
          simp only [hlt, ite_true] at hc
          exact htrace_safe small _ dv b H hc hrows (hsmall hlt)
        · rename_i hlt
          simp only [hlt, ite_false] at hc
          exact htrace_safe big g dv b H hc hrows inv
  | .alloc s body, g, dv, b, H, hc, hrows, inv => by
    simp only [hcheck] at hc
    simp only [htrace]
    by_cases hs : s = slot ∧ allocSize rn slot g body ≠ 0
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hs)] at hc ⊢
      generalize allocSize rn slot g body = n at hc ⊢
      have hid : H (g.depth + 1) = none := inv.top _ (Nat.lt_succ_self _)
      have inv1 : Inv rn { g with store := some (g.depth + 1, n), depth := g.depth + 1 } dv b
          (hset H (g.depth + 1) (some ⟨n, [], none⟩)) :=
        ⟨inv.holds, inv.len, inv.pos, inv.unit, inv.place,
          fun hr => by
            obtain ⟨blk, hb, ht, hsh⟩ := inv.blk hr
            have hne : dv.loc ≠ g.depth + 1 := Nat.ne_of_lt (Nat.lt_succ_of_le (inv.loc hr).1)
            exact ⟨blk, by rw [hset_ne _ _ hne]; exact hb, ht, hsh⟩,
          fun hr => ⟨Nat.le_succ_of_le (inv.loc hr).1, fun sid cap h => by
            cases h
            exact (Nat.ne_of_lt (Nat.lt_succ_of_le (inv.loc hr).1)).symm⟩,
          fun i hi => by
            have hi' : g.depth + 1 < i := hi
            rw [hset_ne _ _ (Nat.ne_of_gt hi')]
            exact inv.top i (Nat.lt_of_succ_lt hi'),
          fun sid cap h => by
            cases h
            exact ⟨Nat.le_refl _, ⟨n, [], none⟩, hset_same _ _ _, rfl⟩⟩
      obtain ⟨H2, e2, f2⟩ := htrace_safe body _ dv b _ hc hrows inv1
      obtain ⟨blk2, hb2, _⟩ := f2.2 (g.depth + 1) n rfl
      have hother : ∀ i, i ≠ g.depth + 1 → H2 i = H i := fun i hi => by
        rw [f2.1 i fun cap h => by cases h; exact hi rfl, hset_ne _ _ hi]
      refine ⟨hset H2 (g.depth + 1) none, ?_, fun i _ => ?_, fun sid cap h => ?_⟩
      · simp only [List.singleton_append, hreplay, hstep, hid, Option.bind_some, hreplay_append, e2, hb2]
      · by_cases h : i = g.depth + 1
        · rw [h, hset_same, hid]
        · rw [hset_ne _ _ h, hother i h]
      · obtain ⟨hle, blk, hb, hcap⟩ := inv.store sid cap h
        have hne : sid ≠ g.depth + 1 := Nat.ne_of_lt (Nat.lt_succ_of_le hle)
        exact ⟨blk, by rw [hset_ne _ _ hne, hother sid hne]; exact hb, hcap⟩
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hs)] at hc ⊢
      exact htrace_safe body g dv b H hc hrows inv
  | .pack d w s dep body, g1, dv, b, H, hc, hrows, inv => by
    simp only [hcheck] at hc
    simp only [htrace]
    have hfr : Frame g1 H = Frame (ownPack s g1) H := rfl
    have hrows : Rows rn (ownPack s g1) := hrows
    have inv := inv.ownPack s
    rw [hfr]
    generalize ownPack s g1 = g at hc hrows inv ⊢
    by_cases hs : s = slot
    · simp only [hs, ite_true, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, beq_iff_eq] at hc ⊢
      obtain ⟨⟨⟨⟨⟨hr, hw⟩, hd⟩, hrd⟩, hdep⟩, hc⟩ := hc
      simp only [hr, ite_true]
      obtain ⟨blk, hb, ht, hsh⟩ := inv.blk hr
      have hreads : ∀ o ∈ points (fillBox rn d (fun k => dep.getD k false) b),
          inShape (dv.idx dv.q (vecOf o)) blk.shape = true := fun o ho =>
        hsh dv.q (vecOf o) fun e he hre => read_panel inv hr (vecOf o) he (.inl (by
          have hde : dep.getD e false = true := (hdep e he).trans hre
          refine Nat.lt_of_lt_of_le (mem_points_range ho e he) (Nat.le_of_eq ?_)
          show (if e = d ∨ dep.getD e false = true then boxSz b e else 1) = boxSz b e
          rw [ite_eq_left_of_eq_true _ _ (eq_true (Or.inr hde))]))
      have hshape : ∀ q o : Vec,
          (∀ e < rn, g.reads e = true →
            (q e * (if e = d then w else 1) + o e) / (if e = d then w else 1) < AffE.up (if e = d then w else 1) (boxSz b e)) →
          inShape (packIdx rn d w (fun k => dep.getD k false) q o) (packShape rn d w (fun k => dep.getD k false) g.a) = true :=
        fun q o h => pack_inShape inv.holds inv.len hw hd hrd hdep q o h
      have hunit : ∀ e, 0 < (if e = d then w else 1) := fun e => by
        by_cases h : e = d
        · rw [ite_eq_left_of_eq_true _ _ (eq_true h)]; exact hw
        · rw [ite_eq_right_of_eq_false _ _ (eq_false h)]; exact Nat.one_pos
      generalize hsp : packShape rn d w (fun k => dep.getD k false) g.a = shape at hc hshape ⊢
      cases hst : g.store with
      | none =>
        rw [hst] at hc
        simp only
        have hid : H (g.depth + 1) = none := inv.top _ (Nat.lt_succ_self _)
        have hne : dv.loc ≠ g.depth + 1 := Nat.ne_of_lt (Nat.lt_succ_of_le (inv.loc hr).1)
        obtain ⟨H', e', f'⟩ := own_block hid (prod shape) shape b _ _ (Nat.le_refl _)
          (hreplay_reads _ dv.loc dv.tag blk (by rw [hset_ne _ _ hne]; exact hb) ht
            (fun o => dv.idx dv.q (vecOf o)) _ hreads)
          (by
            obtain ⟨H3, e3, f3⟩ := htrace_safe body { packG d w g with depth := g.depth + 1 }
              (packD rn d w (fun k => dep.getD k false) (g.depth + 1) b) b
              (hset H (g.depth + 1) (some ⟨prod shape, shape, some b⟩)) hc hrows
              (inv.fresh { packG d w g with depth := g.depth + 1 }
                (packD rn d w (fun k => dep.getD k false) (g.depth + 1) b)
                (hset H (g.depth + 1) (some ⟨prod shape, shape, some b⟩)) rfl hunit rfl (fun _ => rfl) rfl rfl
                ⟨⟨prod shape, shape, some b⟩, hset_same _ _ _, rfl, hshape⟩ (Nat.le_refl _) fun i hi => by
                  have hi' : g.depth + 1 < i := hi
                  rw [hset_ne _ _ (Nat.ne_of_gt hi')]
                  exact inv.top i (Nat.lt_of_succ_lt hi'))
            exact ⟨H3, e3, fun i => f3.1 i fun cap h => by cases h⟩)
        exact ⟨H', e', fun i _ => f' i, fun sid cap h => by rw [hst] at h; cases h⟩
      | some sc =>
        obtain ⟨sid, cap⟩ := sc
        rw [hst] at hc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        simp only
        obtain ⟨hsle, blk0, hb0, hcap0⟩ := inv.store sid cap hst
        obtain ⟨H', e', f', hb'⟩ := store_block hb0 shape b _ _ (hcap0 ▸ hc.1)
          (hreplay_reads H dv.loc dv.tag blk hb ht (fun o => dv.idx dv.q (vecOf o)) _ hreads)
          (by
            obtain ⟨H3, e3, f3⟩ := htrace_safe body (packG d w g) (packD rn d w (fun k => dep.getD k false) sid b) b
              (hset H sid (some ⟨blk0.cap, shape, some b⟩)) hc.2 hrows
              (inv.fresh (packG d w g) (packD rn d w (fun k => dep.getD k false) sid b)
                (hset H sid (some ⟨blk0.cap, shape, some b⟩)) rfl hunit rfl (fun _ => rfl) rfl rfl
                ⟨⟨blk0.cap, shape, some b⟩, hset_same _ _ _, rfl, hshape⟩ hsle fun i hi => by
                  have hi' : g.depth < i := hi
                  rw [hset_ne _ _ (Nat.ne_of_gt (Nat.lt_of_le_of_lt hsle hi'))]
                  exact inv.top i hi')
            exact ⟨H3, e3, fun i => f3.1 i fun cap h => by cases h⟩)
        refine ⟨H', e', fun i hi => f' i fun h => hi cap (by rw [hst, h]), fun sid' cap' h => ?_⟩
        rw [hst] at h
        cases h
        exact hcap0 ▸ hb'
    · simp only [hs, ite_false] at hc ⊢
      exact htrace_safe body g dv b H hc hrows inv
  | .stage s body, g1, dv, b, H, hc, hrows, inv => by
    simp only [hcheck] at hc
    simp only [htrace]
    have hfr : Frame g1 H = Frame (ownStage s g1) H := rfl
    have hrows : Rows rn (ownStage s g1) := hrows
    have inv := inv.ownStage s
    rw [hfr]
    generalize ownStage s g1 = g at hc hrows inv ⊢
    by_cases hs : s = slot ∧ g.ready = false
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hs)] at hc ⊢
      have hshape : ∀ q o : Vec, (∀ e < rn, g.reads e = true → (q e * 1 + o e) / 1 < AffE.up 1 (boxSz b e)) →
          inShape (stageIdx g b q o) (stageShape g b) = true := fun q o h => stage_inShape hrows b q o h
      cases hst : g.store with
      | none =>
        rw [hst] at hc
        simp only
        have hid : H (g.depth + 1) = none := inv.top _ (Nat.lt_succ_self _)
        obtain ⟨H', e', f'⟩ := own_block hid (prod (stageShape g b)) (stageShape g b) b [] _ (Nat.le_refl _) rfl
          (by
            obtain ⟨H3, e3, f3⟩ := htrace_safe body { stageG g with depth := g.depth + 1 } (stageD g (g.depth + 1) b) b
              (hset H (g.depth + 1) (some ⟨prod (stageShape g b), stageShape g b, some b⟩)) hc hrows
              (inv.fresh { stageG g with depth := g.depth + 1 } (stageD g (g.depth + 1) b)
                (hset H (g.depth + 1) (some ⟨prod (stageShape g b), stageShape g b, some b⟩)) rfl
                (fun _ => Nat.one_pos) rfl (fun _ => rfl) rfl rfl
                ⟨⟨prod (stageShape g b), stageShape g b, some b⟩, hset_same _ _ _, rfl, hshape⟩ (Nat.le_refl _)
                fun i hi => by
                  have hi' : g.depth + 1 < i := hi
                  rw [hset_ne _ _ (Nat.ne_of_gt hi')]
                  exact inv.top i (Nat.lt_of_succ_lt hi'))
            exact ⟨H3, e3, fun i => f3.1 i fun cap h => by cases h⟩)
        exact ⟨H', e', fun i _ => f' i, fun sid cap h => by rw [hst] at h; cases h⟩
      | some sc =>
        obtain ⟨sid, cap⟩ := sc
        rw [hst] at hc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        simp only
        obtain ⟨hsle, blk0, hb0, hcap0⟩ := inv.store sid cap hst
        have hfit : prod (stageShape g b) ≤ blk0.cap := by
          rw [hcap0]
          refine Nat.le_trans ?_ hc.1
          refine stage_fits g.vals (boxLo b) (boxSz b) (capOf1 g.a) (fun e => ?_) g.rows fun r hr => (hrows r hr).1
          by_cases he : e < b.length
          · have hea : e < g.a.length := inv.holds.1 ▸ he
            rw [boxSz_get he]
            simp only [capOf1, List.getElem?_eq_getElem hea]
            exact (inv.holds.2 e hea he).1
          · have hea : g.a.length ≤ e := by rw [inv.holds.1]; exact Nat.le_of_not_lt he
            rw [boxSz_ge (Nat.le_of_not_lt he)]
            simp [capOf1, List.getElem?_eq_none hea]
        obtain ⟨H', e', f', hb'⟩ := store_block hb0 (stageShape g b) b [] _ hfit rfl
          (by
            obtain ⟨H3, e3, f3⟩ := htrace_safe body (stageG g) (stageD g sid b) b
              (hset H sid (some ⟨blk0.cap, stageShape g b, some b⟩)) hc.2 hrows
              (inv.fresh (stageG g) (stageD g sid b) (hset H sid (some ⟨blk0.cap, stageShape g b, some b⟩)) rfl
                (fun _ => Nat.one_pos) rfl (fun _ => rfl) rfl rfl
                ⟨⟨blk0.cap, stageShape g b, some b⟩, hset_same _ _ _, rfl, hshape⟩ hsle fun i hi => by
                  have hi' : g.depth < i := hi
                  rw [hset_ne _ _ (Nat.ne_of_gt (Nat.lt_of_le_of_lt hsle hi'))]
                  exact inv.top i hi')
            exact ⟨H3, e3, fun i => f3.1 i fun cap h => by cases h⟩)
        refine ⟨H', e', fun i hi => f' i fun h => hi cap (by rw [hst, h]), fun sid' cap' h => ?_⟩
        rw [hst] at h
        cases h
        exact hcap0 ▸ hb'
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hs)] at hc ⊢
      exact htrace_safe body g dv b H hc hrows inv

/-- The state `run_levels` starts a slot that holds its data in: the view reads the input buffer at address 0 at the global point. -/
def rootG (rows : List AffE) (vals : Nat → Nat) (reads : Nat → Bool) (a : AState) (own fills : List Bool) : HG :=
  { ready := true, unit := fun _ => 1, reads := reads, vals := vals, rows := rows, store := none, depth := 0, a := a,
    own := own, fills := fills }

def rootD (rows : List AffE) (vals : Nat → Nat) (b : Box) : HD :=
  { loc := 0, tag := b, q := boxLo b, idx := fun q o => rows.map fun r => r.eval vals fun e => q e + o e,
    home := fun _ => 0, hsz := fun e => boxLo b e + boxSz b e }

/-- A slot that holds its data: if its access map takes the box into its input buffer, the run is safe and leaves the heap as it was. -/
theorem root_safe {rn ro slot : Nat} (t : Tree) (rows : List AffE) (vals : Nat → Nat) (reads : Nat → Bool) (a : AState)
    (own fills : List Bool) (b : Box) (H : Heap) (shape : List Nat) (hc : hcheck rn ro slot t (rootG rows vals reads a own fills) = true)
    (hrows : Rows rn (rootG rows vals reads a own fills)) (hh : Holds a b) (hlen : b.length = rn) (hpos : ∀ e < rn, 0 < boxSz b e)
    (hH : H 0 = some ⟨prod shape, shape, some b⟩) (htop : ∀ i, 0 < i → H i = none)
    (hin : ∀ p : Vec, (∀ e < rn, reads e = true → p e < boxLo b e + boxSz b e) →
      inShape (rows.map fun r => r.eval vals p) shape = true) :
    ∃ H', hreplay H (htrace rn ro slot t (rootG rows vals reads a own fills) (rootD rows vals b) b) = some H' ∧ ∀ i, H' i = H i := by
  have inv : Inv rn (rootG rows vals reads a own fills) (rootD rows vals b) b H :=
    { holds := hh
      len := hlen
      pos := hpos
      unit := fun _ => Nat.one_pos
      place := fun _ e _ => by simp [rootD, rootG]
      blk := fun _ => ⟨_, hH, rfl, fun q o h => hin (fun e => q e + o e) fun e he hre => by
        have := h e he hre
        simpa only [rootG, rootD, Nat.mul_one, Nat.div_one, up_one] using this⟩
      loc := fun _ => ⟨Nat.le_refl _, fun sid cap h => by cases h⟩
      top := htop
      store := fun sid cap h => by cases h }
  obtain ⟨H', e', f'⟩ := htrace_safe t _ _ b H hc hrows inv
  exact ⟨H', e', fun i => f'.1 i fun cap h => by cases h⟩

/-- A producer slot: the run is safe from a heap without blocks, and leaves none. -/
theorem root_safe_producer {rn ro slot : Nat} (t : Tree) (g : HG) (dv : HD) (b : Box) (H : Heap)
    (hg : g.ready = false) (hst : g.store = none) (hu : ∀ e, 0 < g.unit e) (hc : hcheck rn ro slot t g = true)
    (hrows : Rows rn g) (hh : Holds g.a b) (hlen : b.length = rn) (hpos : ∀ e < rn, 0 < boxSz b e)
    (htop : ∀ i, g.depth < i → H i = none) :
    ∃ H', hreplay H (htrace rn ro slot t g dv b) = some H' ∧ ∀ i, H' i = H i := by
  have inv : Inv rn g dv b H :=
    { holds := hh
      len := hlen
      pos := hpos
      unit := hu
      place := fun hr => by rw [hg] at hr; cases hr
      blk := fun hr => by rw [hg] at hr; cases hr
      loc := fun hr => by rw [hg] at hr; cases hr
      top := htop
      store := fun sid cap h => by rw [hst] at h; cases h }
  obtain ⟨H', e', f'⟩ := htrace_safe t g dv b H hc hrows inv
  exact ⟨H', e', fun i => f'.1 i fun cap h => by rw [hst] at h; cases h⟩

end Sched
