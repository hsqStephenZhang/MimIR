import Sched

/-!
Reads the log of `mim -VVVV` on stdin and runs the checks of the model on every schedule tree `LowerToMem` built.
A tree comes as the line `schedule tree: ro <n> lanes <n> sizes <nats> caps <nats> dyn <bools> fuse <nats> reads <bools>` followed by one node per line.
-/

open Sched

def tokens (s : String) : List String :=
  (s.map fun c => if c = '(' ∨ c = ')' ∨ c = ',' then ' ' else c).splitOn " " |>.filter (· ≠ "")

def nat (s : String) : Nat := (String.ofList (s.toList.filter Char.isDigit)).toNat?.getD 0

def keys : List String := ["ro", "lanes", "sizes", "caps", "dyn", "fuse", "reads"]

/-- The tokens that follow the keyword `k`, up to the next keyword. -/
def field (k : String) (ts : List String) : List String :=
  ((ts.dropWhile (· ≠ k)).drop 1).takeWhile (!keys.contains ·)

structure Line where
  depth : Nat
  name  : String
  args  : List String

def nodes : List String := ["Leaf", "Split", "Reduce", "Pack", "Stage", "Alloc", "Switch"]

def line? (s : String) : Option Line :=
  match tokens s with
  | name :: args =>
    if nodes.contains name then some ⟨(s.toList.takeWhile (· = ' ')).length / 2, name, args⟩ else none
  | [] => none

def mode (s : String) : Mode :=
  if s = "Loop" then .loop else if s = "Peel" then .peel else if s = "Grid" then .grid
  else if s = "Threads" then .threads else if s = "Piped" then .piped else .fixed

def build : Nat → List Line → Option (Tree × List Line)
  | 0, _ => none
  | _, [] => none
  | fuel + 1, l :: ls =>
    let a := fun i => nat (l.args.getD i "0")
    match l.name with
    | "Leaf" => some (.leaf (a 0), ls)
    | "Switch" => do
      let (small, ls) ← build fuel ls
      let (big, ls) ← build fuel ls
      pure (.switch (a 0) (a 1) small big, ls)
    | name => do
      let (body, ls) ← build fuel ls
      match name with
      | "Split" => pure (.split (a 0) (a 1) (mode (l.args.getD 2 "")) body, ls)
      | "Reduce" => pure (.reduce (a 0) (a 1) body, ls)
      | "Pack" => pure (.pack (a 0) (a 1) (a 2) ((l.args.drop 3).map (· == "tt")) body, ls)
      | "Stage" => pure (.stage (a 0) body, ls)
      | "Alloc" => pure (.alloc (a 0) body, ls)
      | _ => none

def chunks (n : Nat) (l : List Bool) : Nat → List (List Bool)
  | 0 => []
  | k + 1 => l.take n :: chunks n (l.drop n) k

/-- Runs the checks on one tree and tells which of them fails. -/
def verdict (header : List String) (t : Tree) : String :=
  let ro := nat ((field "ro" header).getD 0 "0")
  let lanes := nat ((field "lanes" header).getD 0 "0")
  let sizes := (field "sizes" header).map nat
  let caps := (field "caps" header).map nat
  let dyn : List Bool := (field "dyn" header).map (· == "tt")
  let fuse := (field "fuse" header).map nat
  let rn := sizes.length
  let reads := chunks rn ((field "reads" header).map (· == "tt")) fuse.length
  let a : AState := (caps.zip dyn).map fun (p : Nat × Bool) => ⟨p.1, p.2, false, 0⟩
  let gs : List Ghost := (fuse.zip reads).map fun (p : Nat × List Bool) => ⟨p.1 = 0, fun _ => 1, fun k => p.2.getD k false⟩
  if !t.wf rn ro then "FAIL wf"
  else if !scheck lanes ro t a then "FAIL scheck"
  else if !vcheck rn t gs then "FAIL vcheck"
  else if !echeck ro t a gs then "ok, but an edge tile may read beyond a panel"
  else if !(List.range fuse.length).all (fun i =>
      hcheck rn ro i t ⟨fuse.getD i 0 == 0, fun _ => 1, fun k => (reads.getD i []).getD k false, fun _ => 0, [], none, 0, a,
        fuse.map fun _ => true, fuse.map (· != 0)⟩)
    then "ok, but the checks of the heap machine fail"
  else "ok"

partial def scan (seen bad : Nat) : List String → IO (Nat × Nat)
  | [] => pure (seen, bad)
  | l :: ls =>
    match (l.splitOn "schedule tree: ").drop 1 with
    | h :: _ => do
      let body := ls.takeWhile fun s => (line? s).isSome
      let rest := ls.drop body.length
      let lines := body.filterMap line?
      match build (lines.length + 1) lines with
      | some (t, []) =>
        let v := verdict (tokens h) t
        IO.println s!"{v}: {h}"
        scan (seen + 1) (if v.startsWith "ok" then bad else bad + 1) rest
      | _ =>
        IO.println s!"FAIL parse: {h}"
        scan (seen + 1) (bad + 1) rest
    | [] => scan seen bad ls

def main : IO UInt32 := do
  let stdin ← IO.getStdin
  let mut lines := #[]
  repeat
    let l ← stdin.getLine
    if l.isEmpty then break
    lines := lines.push ((l.splitOn "\n").headD "")
  let (seen, bad) ← scan 0 0 lines.toList
  IO.println s!"{seen} trees, {bad} rejected"
  pure (if bad = 0 then 0 else 1)
