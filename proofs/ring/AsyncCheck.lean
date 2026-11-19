import Ring.AsyncCode

open Ring.Async

private def parse (ws : List String) : Option Instr :=
  match ws with
  | ["fill", j, s, src] => do pure (.fill (← j.toNat?) (← s.toNat?) (← src.toNat?))
  | ["commit"] => some .commit
  | ["wait", n] => do pure (.waitGroup (← n.toNat?))
  | ["proxy_fence"] => some .proxyFence
  | ["bar"] => some .barrier
  | ["read", b, s, "load"] => do pure (.read (← b.toNat?) (← s.toNat?) .load)
  | ["read", b, s, "wgmma", c] => do pure (.read (← b.toNat?) (← s.toNat?) (.wgmma (← c.toNat?)))
  | ["wgmma_fence", c] => do pure (.wgmmaFence (← c.toNat?))
  | ["wgmma_commit", c] => do pure (.wgmmaCommit (← c.toNat?))
  | ["wgmma_wait0", c] => do pure (.wgmmaWait0 (← c.toNat?))
  | ["settle"] => some .settle
  | _ => none

private def render : Instr → String
  | .fill j s src => s!"fill {j} {s} {src}"
  | .commit => "commit"
  | .waitGroup n => s!"wait {n}"
  | .proxyFence => "proxy_fence"
  | .barrier => "bar"
  | .read b s .load => s!"read {b} {s} load"
  | .read b s (.wgmma c) => s!"read {b} {s} wgmma {c}"
  | .wgmmaFence c => s!"wgmma_fence {c}"
  | .wgmmaCommit c => s!"wgmma_commit {c}"
  | .wgmmaWait0 c => s!"wgmma_wait0 {c}"
  | .settle => "settle"

private def parseProxy : String → Option Bool
  | "0" | "false" => some false
  | "1" | "true" => some true
  | _ => none

def main (args : List String) : IO UInt32 := do
  if let ["--expected", d, count, proxy, loads, mmas] := args then
    if let (some d, some count, some proxy, some loads, some mmas) :=
        (d.toNat?, count.toNat?, parseProxy proxy, loads.toNat?, mmas.toNat?) then
      for op in expected d count proxy loads mmas do IO.println (render op)
      return 0
  let [d, count, proxy] := args
    | IO.eprintln "usage: ring_async_check DEPTH COUNT PROXY(0|1) < ops"; return 2
  let (some d, some count, some proxy) := (d.toNat?, count.toNat?, parseProxy proxy)
    | IO.eprintln "usage: ring_async_check DEPTH COUNT PROXY(0|1) < ops"; return 2
  let text ← (← IO.getStdin).readToEnd
  let mut ops : Array (Nat × String × Instr) := #[]
  let mut line := 0
  for raw in text.splitOn "\n" do
    line := line + 1
    let body := ((raw.splitOn "#").headD "").trimAscii.toString
    if body.isEmpty then continue
    let ws := (body.splitOn " ").filter (· ≠ "")
    match parse ws with
    | some op => ops := ops.push (line, body, op)
    | none => IO.eprintln s!"line {line}: cannot parse '{body}'"; return 2
  match diagnose d count proxy (ops.toList.map (·.2.2)) with
  | none => IO.println "accepted"; return 0
  | some (i, why) =>
    match ops[i]? with
    | some (l, body, _) => IO.println s!"rejected op {i} (line {l}) '{body}': {why}"
    | none => IO.println s!"rejected at end (op {i}): {why}"
    return 1
