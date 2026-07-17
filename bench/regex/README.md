# Regex Cross-Engine Comparison

This harness compares the common semantic subset of:

- Python `RegBuilder`, which builds `%regex.*` and uses `LowerRegex`;
- native `.mim` `%regex.*`, which uses the same `LowerRegex` pass without
  constructing the regex through Python;
- Rust `regex::bytes::Regex` from a local rust-lang/regex checkout;
- Rust `regex::bytes::Regex` built with `RUSTFLAGS=-C target-cpu=native`;
- the closed-DFA Futamura specialization in `%regex.dfa.compile`.

Only ASCII byte patterns with full-string matching are included. Rust patterns
are wrapped in `\A(?:...)\z`; Python `RegBuilder` checks that the matcher stops
at NUL; and the closed DFA accepts only when NUL is seen in an accepting state.
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

The Futamura PE column is fed by a small benchmark-local Thompson NFA -> DFA
generator. It emits closed `%regex.DFA` tables and then lets
`%regex.dfa.compile` specialize them, so the POPL'25 e-mail regex is covered as
well. The generator is intentionally limited to the shared benchmark subset:
literal bytes, byte ranges, concatenation, alternation, `?`, `*`, and `+`.

Build/setup times are printed separately and are not directly comparable:
Rust Cargo incremental state, MimIR plugin loading, and Python's per-pattern
JIT organization differ. The steady-state table reports median nanoseconds per
match and MiB/s for short, medium, approximately 4 KiB, and approximately
64 KiB haystacks.

The rust-lang/regex repository itself no longer contains an active benchmark
suite; its `bench/README.md` points to the external Rebar project. This harness
is intentionally smaller and focused on semantics MimIR currently shares.
