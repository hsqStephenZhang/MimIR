import Ring.Fold

open Ring

/-- Observable operands in the one-panel fixture. Panel and barrier offsets are in
    separate arrays. A successful poll identifies its slot and the parity it tested. -/
private def events (d : Nat) : Op → List String
  | .fill j slot =>
    [s!"expect {8 * slot} 128", s!"fill {j} {128 * slot} {8 * slot}"]
  | .poll i slot => [s!"poll {8 * slot} {i / d % 2}"]
  | .release none => ["release 0 0"]
  | .release (some i) => [s!"release 1 {i % d}"]
  | .mma i slot => [s!"mma {i} {128 * slot}"]
  | .commit => ["commit"]
  | .waitGroups n => [s!"wait_group {n}"]
  | .next | .fence | .wgmmaFence => []

def main (args : List String) : IO Unit := do
  let [some d, some count, some lag, some width] := args.map String.toNat?
    | throw (IO.userError "usage: ring_trace DEPTH COUNT LAG WIDTH")
  if d == 0 || lag > 1 then throw (IO.userError "expected positive depth and lag 0 or 1")
  for s in List.range d do IO.println s!"init {8 * s} 1"
  let ops := fold d count (if lag == 0 then .zero else .one) (fun _ => width)
  for op in ops do
    for line in events d op do IO.println line
