# Regex Cross-Engine Comparison

This harness compares the common semantic subset of:

- Python `RegBuilder`, which builds `%regex.*` and uses `LowerRegex`;
- native `.mim` `%regex.*`, which uses the same `LowerRegex` pass without
  constructing the regex through Python;
- Rust `regex::bytes::Regex` from a local rust-lang/regex checkout;
- Rust `regex::bytes::Regex` built with `RUSTFLAGS=-C target-cpu=native`;
- the Graal-style MimIR baseline, which writes `%regex.*`, lets the host pass
  build a closed `%regex.DFA`, and specializes it with `%regex.specialize_dfa`.

Only ASCII byte patterns with full-string matching are included. Rust patterns
are wrapped in `\A(?:...)\z`; Python `RegBuilder` checks that the matcher stops
at NUL; and the Graal-style DFA accepts only when NUL is seen in an accepting
state.
Captures, leftmost-first submatch selection, Unicode, word boundaries,
look-around, and replacement semantics are intentionally excluded.

Run correctness only:

```sh
.venv/bin/python bench/regex/compare.py --check-only
```

Run the benchmark:

```sh
.venv/bin/python bench/regex/compare.py \
  --rust-regex-dir /workspaces/rust/regex \
  --iterations 3000 \
  --samples 5
```

The harness generates temporary shared libraries for all engines. Each
exports a batch loop, so Python iteration and per-match ctypes overhead are not
included. Python/PE call a generated matcher through a C function call; the
Rust matchers are marked `inline(never)` to keep a similar call boundary. The C
shim calls generated MimIR matchers through a volatile function pointer, and the
Rust loops use `std::hint::black_box`, so the batch loop should not fold a closed
input into one constant result. The `rust-regex-bytes-native` column uses the
same Rust source and only changes code generation with `-C target-cpu=native`.

The current pattern set covers:

- literal concatenation: `abc`;
- one-or-more repetition: `a+b`, `(abc)+`;
- ASCII byte classes/ranges: `[A-Za-z0-9_]+`, `[0-9]+`;
- identifier-like composition: `[A-Za-z_][A-Za-z0-9_]*`;
- optional branch: `colou?r`;
- short keyword alternation: `cat|car|dog`;
- alternation from rust-lang/regex Fowler `basic26`: `(ab|a)(bc|c)`;
- the deterministic e-mail regex used by the POPL'25 artifact regex
  evaluation.

In addition to per-pattern microbenchmarks, the harness can run the e-mail
regex over the POPL'25 artifact corpus:

```sh
.venv/bin/python bench/regex/compare.py \
  --paper-email-corpus /workspaces/compile/popl25/mimir_regex_benchmark/fradulent_emails.txt
```

When the source `fradulent_emails.txt` file is used, the harness performs the
same address extraction and annotation step as the artifact script: extract
`[a-zA-Z0-9_.\-]+@[a-zA-Z0-9_.\-]+`, deduplicate, and label each address with
the deterministic e-mail regex. In the local artifact checkout this yields
10,663 addresses: 10,215 accepted and 448 rejected. The corpus loop is emitted
into the native C/Rust shims, so the Python harness does not call the matcher
once per address.

The `graal-style-pe` column is fed by `bench/regex/futamura.mim`, but that file
now contains high-level `%regex.*` expressions rather than hand-written DFA
tables. `%regex.host.compile` marks the Graal-style boundary: C++ constructs
NFA/DFA on the host, materializes a closed `%regex.DFA`, and then reuses the
closed-table specialization path. The stronger Futamura route, where
RegexExpr -> NFA -> DFA is itself implemented in MimIR and erased by partial
evaluation, remains separate from this benchmark baseline.

Build/setup times are printed separately and are not directly comparable:
Rust Cargo incremental state, MimIR plugin loading, and Python's per-pattern
JIT organization differ. The steady-state table reports median nanoseconds per
match and MiB/s for short, medium, approximately 4 KiB, and approximately
64 KiB haystacks.

The rust-lang/regex repository itself no longer contains an active benchmark
suite; its `bench/README.md` points to the external Rebar project. This harness
is intentionally smaller and focused on semantics MimIR currently shares.

## Latest Local Results

Measured on 2026-07-17 with:

- Machine: `rusty`
- OS: Linux `6.12.85-1-MANJARO` x86_64
- CPU: Intel(R) Core(TM) i5-14600KF
- CPU topology: 14 cores, 20 hardware threads
- Command:

```sh
python3 bench/regex/compare.py \
  --iterations 3000 \
  --samples 5 \
  --no-paper-email-corpus
```

The table reports median `ns/match`. The paper corpus loop is intentionally
excluded here; the last two `paper_email` rows are the small built-in valid and
invalid examples.

Build/setup time:

| Engine | Time |
|---|---:|
| python-lower-regex | 24.509 s |
| native-lower-regex | 3.649 s |
| rust-regex-bytes | 2.775 s |
| rust-regex-bytes-native | 2.945 s |
| graal-style-pe | 3.615 s |

Representative steady-state results:

| Pattern | Workload | Bytes | python-lower-regex | native-lower-regex | rust-regex-bytes | rust-regex-bytes-native | graal-style-pe |
|---|---|---:|---:|---:|---:|---:|---:|
| literal_abc | short-match | 3 | 6.2 | 5.3 | 73.4 | 75.3 | 4.3 |
| literal_abc | short-reject | 3 | 6.1 | 5.2 | 13.7 | 14.1 | 1.3 |
| a_plus_b | medium-match | 256 | 56.4 | 56.5 | 303.2 | 302.1 | 56.4 |
| a_plus_b | huge-match | 65536 | 12428.6 | 12442.3 | 74444.1 | 74461.5 | 12438.4 |
| abc_plus | medium-match | 255 | 393.2 | 392.9 | 300.9 | 301.9 | 40.6 |
| abc_plus | huge-match | 65535 | 99969.3 | 100053.1 | 74473.5 | 74461.7 | 8311.1 |
| ascii_word_plus | medium-match | 256 | 683.8 | 685.6 | 302.2 | 302.3 | 115.8 |
| ascii_word_plus | huge-match | 65536 | 174899.6 | 174865.9 | 74434.0 | 74418.9 | 27601.3 |
| digit_plus | medium-match | 256 | 57.3 | 57.0 | 302.2 | 302.1 | 85.3 |
| digit_plus | huge-match | 65536 | 12435.7 | 12440.1 | 74431.9 | 74439.6 | 23486.1 |
| identifier | medium-match | 256 | 684.4 | 685.3 | 302.4 | 302.4 | 113.0 |
| identifier | huge-match | 65536 | 174794.0 | 174954.6 | 74435.3 | 74442.1 | 27205.3 |
| color_optional | short-match | 6 | 2.0 | 2.2 | 18.3 | 18.8 | 1.3 |
| keyword_alt | short-match | 3 | 1.5 | 1.6 | 15.3 | 15.8 | 1.5 |
| fowler_basic26 | short-match | 4 | 1.5 | 1.3 | 15.7 | 16.2 | 1.5 |
| paper_email | paper-valid | 34 | 26.6 | 26.5 | 50.0 | 50.7 | 22.8 |
| paper_email | paper-invalid | 15 | 12.5 | 11.4 | 28.1 | 28.2 | 8.7 |
