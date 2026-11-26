import Sched.HSafe

/-!
# The rejections of `lit/tensor/pe_cpu_sched_check.mim`

Each tree that `tensor.mim` rejects there with a `SchedTree:` message is rejected by the checks of the model, and the trees it accepts next to them are accepted.
-/

namespace Sched

def axes (caps : List Nat) (dyn : List Bool) : AState := (caps.zip dyn).map fun p => ⟨p.1, p.2, false, 0⟩

def slot (ready : Bool) (reads : List Bool) : Ghost := ⟨ready, fun _ => 1, fun k => reads.getD k false⟩

def accepts (lanes ro : Nat) (t : Tree) (a : AState) (gs : List Ghost) : Bool :=
  check lanes ro t a && vcheck a.length t gs

def static3 (caps : List Nat) : AState := axes caps [false, false, false]
def gemm : List Ghost := [slot true [true, false, true], slot true [false, true, true]]
def fused : List Ghost := [slot false [true, false, true], slot true [false, true, true]]
def sum3 : List Ghost := [slot true [true, true, true]]

/-- a Split must cut a packed axis into whole panels -/
example : accepts 0 2 (.pack 1 8 1 [false, true, true] (.split 1 3 .fixed (.leaf 0))) (static3 [4, 16, 4]) gemm = false := by
  decide

/-- a Split needs a positive tile size -/
example : accepts 0 2 (.split 2 0 .loop (.leaf 0)) (static3 [4, 16, 4]) gemm = false := by decide

/-- a Fixed Split needs an axis with a static extent -/
example : accepts 0 2 (.split 0 6 .loop (.split 0 4 .fixed (.leaf 1))) (static3 [10, 8, 4]) gemm = false := by decide

/-- a Leaf reads a producer slot that no Stage above it computes -/
example : accepts 0 2 (.leaf 0) (static3 [4, 16, 4]) fused = false := by decide

example : accepts 0 2 (.stage 0 (.leaf 0)) (static3 [4, 16, 4]) fused = true := by decide

/-- a Pack reads a producer slot that no Stage above it computes -/
example : accepts 0 2 (.pack 0 2 0 [true, false, true] (.stage 0 (.leaf 0))) (static3 [4, 16, 4]) fused = false := by
  decide

/-- a Pack's dep must be the loops its slot reads -/
example : accepts 0 2 (.pack 1 8 1 [true, true, true] (.leaf 0)) (static3 [4, 16, 4]) gemm = false := by decide

/-- a Leaf vectorizes a loop below Rn, or none with Rn -/
example : accepts 0 2 (.leaf 7) (static3 [4, 16, 4]) gemm = false := by decide

/-- only the first reduction loop may be split when the combiner has no merge -/
example : accepts 0 1 (.split 2 4 .fixed (.leaf 3)) (static3 [4, 6, 8]) sum3 = false := by decide

example : accepts 0 1 (.split 1 4 .fixed (.leaf 3)) (static3 [4, 6, 8]) sum3 = true := by decide

/-- a Pack sizes its panel by a run-time extent that no Split bounds -/
example : accepts 0 2 (.pack 0 2 0 [true, false, true] (.leaf 3)) (axes [unbounded, 16, 4] [true, false, false]) gemm =
    false := by decide

/-- a Leaf computes an edge tile of a run-time extent that no Split bounds -/
example : accepts 0 2 (.leaf 0) (axes [unbounded, 16, 4] [true, false, false]) gemm = false := by decide

/-- a Reduce needs a reduction loop -/
example : accepts 0 2 (.reduce 0 4 (.leaf 0)) (static3 [4, 16, 4]) gemm = false := by decide

/-- a Reduce needs at least two partial results -/
example : accepts 0 2 (.reduce 2 1 (.leaf 2)) (static3 [4, 16, 4]) gemm = false := by decide

/-- a Reduce must split the loop its Leaf vectorizes -/
example : accepts 0 2 (.reduce 2 4 (.leaf 1)) (static3 [4, 16, 4]) gemm = false := by decide

/-- a Reduce needs a combiner with a merge -/
example : accepts 0 2 (.reduce 2 4 (.leaf 2)) (static3 [4, 16, 4]) gemm = false := by decide

example : accepts 4 2 (.reduce 2 4 (.leaf 2)) (static3 [4, 16, 4]) gemm = true := by decide

/-- The heap machine's checks for one slot of the contraction. -/
def heapAccepts (slot : Nat) (t : Tree) (a : AState) (reads : List Bool) : Bool :=
  hcheck a.length 2 slot t (rootG [] (fun _ => 0) (fun k => reads.getD k false) a [true, true] [false, false])

/-- a Pack needs a loop its slot reads -/
example : heapAccepts 0 (.pack 1 2 0 [true, false, true] (.leaf 3)) (static3 [4, 16, 4]) [true, false, true] = false := by
  decide

/-- a Leaf computes an edge tile wider than a panel of a slot it reads along a run-time loop -/
example : heapAccepts 0
    (.split 0 6 .loop (.pack 0 2 0 [true, false, true] (.pack 1 8 1 [false, true, true] (.leaf 0))))
    (axes [unbounded, 16, 4] [true, false, false]) [true, false, true] = false := by decide

end Sched
