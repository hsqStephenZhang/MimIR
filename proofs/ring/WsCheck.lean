import Ring.WsCode

open Ring.Tma Ring.WarpSpecialized

private def usage : String :=
  "usage: ring_ws_check DEPTH COUNT < roles\n" ++
  "       ring_ws_check --expected DEPTH COUNT LOADS BYTES MMAS"

private def roleOf : List String → Option Role
  | ["producer"] => some .producer
  | ["consumer", "0"] => some (.consumer true)
  | ["consumer", "1"] => some (.consumer false)
  | _ => none

def main (args : List String) : IO UInt32 := do
  if let ["--expected", d, count, loads, bytes, mmas] := args then
    if let (some d, some count, some loads, some bytes, some mmas) :=
        (d.toNat?, count.toNat?, loads.toNat?, bytes.toNat?, mmas.toNat?) then
      for (name, role) in [("producer", Role.producer), ("consumer 0", .consumer true), ("consumer 1", .consumer false)] do
        IO.println s!"role {name}"
        for op in expected d count role loads bytes mmas do IO.println (render op)
      return 0
  let [d, count] := args | IO.eprintln usage; return 2
  let (some d, some count) := (d.toNat?, count.toNat?) | IO.eprintln usage; return 2
  let roles ← match sections (← (← IO.getStdin).readToEnd) with
    | .ok roles => pure roles
    | .error msg => IO.eprintln msg; return 2
  let mut seen : List (List String) := []
  for role in roles do
    if (roleOf role.header).isNone || seen.contains role.header then
      IO.eprintln s!"line {role.line}: expected each of 'role producer', 'role consumer 0', 'role consumer 1' once"
      return 2
    seen := role.header :: seen
  if seen.length ≠ 3 then
    IO.eprintln "expected the roles producer, consumer 0 and consumer 1"
    return 2
  let mut failed := false
  for role in roles do
    let some r := roleOf role.header | return 2
    let name := " ".intercalate role.header
    match diagnoseRole d count r (role.ops.toList.map (·.2.2)) with
    | none => pure ()
    | some (i, why) =>
      failed := true
      match role.ops[i]? with
      | some (l, body, _) => IO.println s!"rejected role {name} op {i} (line {l}) '{body}': {why}"
      | none => IO.println s!"rejected role {name} at end (op {i}): {why}"
  if failed then return 1
  IO.println "accepted"
  return 0
