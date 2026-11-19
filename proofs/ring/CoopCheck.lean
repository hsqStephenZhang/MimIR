import Ring.CoopCode

open Ring.Tma Ring.Cooperative

private def usage : String :=
  "usage: ring_coop_check DEPTH COUNT < roles\n" ++
  "       ring_coop_check --expected DEPTH COUNT GROUPS LOADS BYTES MMAS"

def main (args : List String) : IO UInt32 := do
  if let ["--expected", d, count, groups, loads, bytes, mmas] := args then
    if let (some d, some count, some groups, some loads, some bytes, some mmas) :=
        (d.toNat?, count.toNat?, groups.toNat?, loads.toNat?, bytes.toNat?, mmas.toNat?) then
      for g in List.range groups do
        IO.println (if g = 0 then "role leader" else s!"role consumer {g}")
        for op in expected d count (g = 0) loads bytes mmas do IO.println (render op)
      return 0
  let [d, count] := args | IO.eprintln usage; return 2
  let (some d, some count) := (d.toNat?, count.toNat?) | IO.eprintln usage; return 2
  let roles ← match sections (← (← IO.getStdin).readToEnd) with
    | .ok roles => pure roles
    | .error msg => IO.eprintln msg; return 2
  if roles.isEmpty then IO.eprintln "no role"; return 2
  let mut failed := false
  for h : g in [0:roles.size] do
    let role := roles[g]
    let want := if g = 0 then ["leader"] else ["consumer", toString g]
    if role.header ≠ want then
      IO.eprintln s!"line {role.line}: expected 'role {" ".intercalate want}'"
      return 2
    let ops := role.ops
    match diagnoseRole d count (g = 0) (ops.toList.map (·.2.2)) with
    | none => pure ()
    | some (i, why) =>
      failed := true
      match ops[i]? with
      | some (l, body, _) => IO.println s!"rejected role {" ".intercalate want} op {i} (line {l}) '{body}': {why}"
      | none => IO.println s!"rejected role {" ".intercalate want} at end (op {i}): {why}"
  if failed then return 1
  IO.println "accepted"
  return 0
