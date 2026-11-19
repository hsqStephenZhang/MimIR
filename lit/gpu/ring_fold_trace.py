"""Run specialized Mim control flow on the host and compare its observations with Ring.fold."""
import difflib
import itertools
import os
from pathlib import Path
import re
import subprocess
import sys


def compile_cases(mim, source, work, depth, cases):
    fixture = source.read_text()
    calls = []
    for rid, (count, lag, turns) in enumerate(cases):
        name = f"fold_{depth}_{count}_{lag}_{int(turns)}"
        mode = "tt" if turns else "ff"
        fixture += (f"\nextern fun {name} (m: M): M = "
                    f"return (run ({rid}, {depth}, {count}, {lag}, {mode}) (m, ⊥:(Bars {depth}), ⊥:Map));\n")
        calls.append(f'extern void {name}(void); puts("{name}"); {name}();')

    (work / "fixture.mim").write_text(fixture)
    subprocess.run([mim, "--werror", "fixture.mim", "-o", "residual.mim"], cwd=work, check=True)
    residual = (work / "residual.mim").read_text()
    if primitive := re.search(r"\bgpu\.\w+", residual):
        raise ValueError(f"uninterpreted device primitive {primitive[0]} in {work}/residual.mim")

    # Bind the opaque sink. Keep all operands, guards, continuations and loop state from PE.
    host, n = re.subn(r"^axm ring_event: [^;]+;$",
                      "extern fun ring_event [mem.M 0, Nat, Nat, Nat, Nat]: mem.M 0;",
                      residual, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"expected one ring_event declaration in {work}/residual.mim")
    host += """
extern lam _compile (): compile.Phase = compile.phases ff (
    compile.named "cps.conv", compile.named "affine.lower_for", compile.scalarize, compile.scalarize,
    compile.lam_spec, compile.single_erasure, compile.dispatch_normalize, compile.ret_wrap, compile.named "ll.emit");
"""
    # Erasure leaves a ring's token and state as units, which the dump may print under one name.
    host = host.replace("v: [], v: []", "_: [], _: []")
    (work / "host.mim").write_text(host)
    subprocess.run([mim, "--werror", "host.mim", "-p", "ll", "-X", "ll:rt=extern", "-o", os.devnull], cwd=work, check=True)
    (work / "main.c").write_text('#include <stdio.h>\nint main(void) {\n' + "\n".join(calls) + "\n}\n")
    subprocess.run(["clang", "-Wno-override-module", "host.ll", source.with_suffix(".c"), "main.c", "-o", "run"],
                   cwd=work, check=True)
    output = subprocess.check_output([work / "run"], text=True, timeout=30)
    return output.splitlines()


def main():
    mim, fixture, proofs, temp = sys.argv[1:]
    source = Path(fixture).resolve()
    proofs = Path(proofs).resolve()
    subprocess.run(["lake", "build", "ring_trace"], cwd=proofs, check=True)
    lean = proofs / ".lake/build/bin/ring_trace"
    equal = missing_wait = 0

    for depth in (2, 3, 4):
        counts = sorted({0, 1, depth - 1, depth, 2 * depth + 1})
        cases = list(itertools.product(counts, (0, 1), (False, True)))
        work = Path(temp).resolve() / str(depth)
        work.mkdir(parents=True, exist_ok=True)
        output = compile_cases(mim, source, work, depth, cases)
        traces = {}
        for line in output:
            if line.startswith("fold_"):
                if line in traces:
                    raise ValueError(f"repeated case {line}")
                current = traces[line] = []
            else:
                current.append(line)

        for count, lag in itertools.product(counts, (0, 1)):
            expected = subprocess.check_output([lean, str(depth), str(count), str(lag), "2"], text=True).splitlines()
            name = f"fold_{depth}_{count}_{lag}"
            rolled, unrolled = (traces.pop(f"{name}_{turns}") for turns in (0, 1))
            if rolled != unrolled:
                sys.exit("\n".join(difflib.unified_diff(rolled, unrolled, name + " rolled", name + " unrolled", lineterm="")))
            if rolled == expected:
                equal += 2
            elif lag == 0 and expected[-1:] == ["wait_group 0"] and rolled == expected[:-1]:
                missing_wait += 2
            else:
                sys.exit("\n".join(difflib.unified_diff(expected, rolled, "Ring.fold", name, lineterm="")))
        if traces:
            raise ValueError(f"unexpected cases {list(traces)}")
    print(f"{equal} exact matches; {missing_wait} traces lack only the model's final wait_group 0")


if __name__ == "__main__":
    main()
