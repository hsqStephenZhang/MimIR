import Sched.HSafe

/-! Runs of the machine on the heap for slot `A` of a contraction over `(m, n, k)`, which reads `(m, k)`. -/

namespace Sched

def rootHeap (shape : List Nat) (b : Box) : Heap := fun i => if i = 0 then some ⟨prod shape, shape, some b⟩ else none

def safe (ro : Nat) (t : Tree) (a : AState) (shape : List Nat) (rows : List AffE) (reads : List Bool) (b : Box) : Bool :=
  (hreplay (rootHeap shape b) (htrace b.length ro 0 t (rootG rows (fun _ => 0) (fun k => reads.getD k false) a [true] [false]) (rootD rows (fun _ => 0) b) b)).isSome

def checked (ro : Nat) (t : Tree) (a : AState) (rows : List AffE) (reads : List Bool) : Bool :=
  hcheck a.length ro 0 t (rootG rows (fun _ => 0) (fun k => reads.getD k false) a [true] [false])

def boxOf (sizes : List Nat) : Box := sizes.map fun n => (0, n)
def aOf (caps : List Nat) (dyn : List Bool) : AState := (caps.zip dyn).map fun p => ⟨p.1, p.2, false, 0⟩
def mk : List AffE := [.dim 0, .dim 2]
def rd : List Bool := [true, false, true]

-- A static contraction with a packed `A`.
#guard safe 2 (.split 2 4 .fixed (.split 0 4 .fixed (.pack 0 2 0 rd (.split 0 2 .fixed (.split 1 2 .fixed (.leaf 0))))))
    (aOf [6, 4, 6] [false, false, false]) [6, 6] mk rd (boxOf [6, 4, 6])

-- A run-time `m` of 6 under a bound of 8: the edge tile of 2 reads 4 rows of its panel.
#guard safe 2 (.split 0 4 .loop (.pack 0 4 0 rd (.leaf 0)))
    (aOf [8, 4, 6] [true, false, false]) [6, 6] mk rd (boxOf [6, 4, 6])

-- Without the `Pack`, the slot reads its own buffer, and the edge tile only runs over its 2 rows.
#guard safe 2 (.split 0 4 .loop (.leaf 0))
    (aOf [8, 4, 6] [true, false, false]) [6, 6] mk rd (boxOf [6, 4, 6])

-- A panel narrower than the edge tile's `cap` does not help: the tile of 1 reads 6 rows, into the panel after the spare one.
-- `hcheck` rejects this tree.
#guard !safe 2 (.split 0 7 .loop (.pack 0 2 0 rd (.split 0 6 .loop (.leaf 0))))
    (aOf [7, 4, 6] [true, false, false]) [7, 6] mk rd (boxOf [7, 4, 6])

-- An `Alloc` whose storage the tiles of `k` fill in turn.
#guard safe 2 (.alloc 0 (.split 2 3 .fixed (.pack 0 2 0 rd (.split 0 2 .fixed (.leaf 0)))))
    (aOf [6, 4, 6] [false, false, false]) [6, 6] mk rd (boxOf [6, 4, 6])

-- The static checks agree with the runs above.
#guard checked 2 (.split 2 4 .fixed (.split 0 4 .fixed (.pack 0 2 0 rd (.split 0 2 .fixed (.split 1 2 .fixed (.leaf 0))))))
    (aOf [6, 4, 6] [false, false, false]) mk rd
#guard checked 2 (.split 0 4 .loop (.pack 0 4 0 rd (.leaf 0))) (aOf [8, 4, 6] [true, false, false]) mk rd
#guard checked 2 (.split 0 4 .loop (.leaf 0)) (aOf [8, 4, 6] [true, false, false]) mk rd
#guard !checked 2 (.split 0 7 .loop (.pack 0 2 0 rd (.split 0 6 .loop (.leaf 0)))) (aOf [7, 4, 6] [true, false, false]) mk rd
#guard checked 2 (.alloc 0 (.split 2 3 .fixed (.pack 0 2 0 rd (.split 0 2 .fixed (.leaf 0)))))
    (aOf [6, 4, 6] [false, false, false]) mk rd

end Sched
