import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
assert len(rows) == 3, rows
by_shape = {tuple(row["So"]): row for row in rows}
ssa = by_shape[(12, 16)]
assert ssa["nest"] == "register_blocked", ssa
assert ssa["register_block"] == [4, 16], ssa
assert ssa["Sr"] == [3, 4, 16, 2], ssa
assert ssa["Ro"] == 2 and ssa["Rp"] == 3 and ssa["Rr"] == 1, ssa
assert ssa["parallel_tasks"] == 3, ssa
fallback = by_shape[(48, 16)]
assert fallback["nest"] == "blocked", fallback
assert fallback["register_block"] is None, fallback
assert fallback["parallel_tasks"] == 0, fallback
classic = by_shape[(3, 7)]
assert classic["nest"] == "classic", classic
assert classic["schedule"]["par"] == 1 and classic["parallel_tasks"] == 0, classic
assert len({r["op_id"] for r in rows}) == 3, rows
for row in rows:
    assert row["schema"] == 1 and row["backend"] == "cpu", row
    assert row["target_threads"] == 4 and row["runtime_threads"] is None, row
    assert not row["predicate"] and not row["staged_producer"], row
