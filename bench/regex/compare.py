#!/usr/bin/env python3
r"""Correctness and runtime comparison for MimIR regex staging paths.

The comparison is deliberately limited to ASCII byte regexes with full-match
semantics. Rust uses ``regex::bytes::Regex`` with ``\A(?:...)\z`` anchors;
Python ``RegBuilder`` adds its existing end-of-input check; and the
Graal-style host-staged path accepts only when NUL is observed in an accepting
DFA state.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
PY_STAGE = ROOT / "build" / "mim_py_stage" / "main" / "src"
PLUGIN_STAGE = ROOT / "build" / "lib" / "mim"
if PY_STAGE.is_dir():
    import sys

    sys.path.insert(0, str(PY_STAGE))
if PLUGIN_STAGE.is_dir():
    old_plugin_path = os.environ.get("MIM_PLUGIN_PATH")
    os.environ["MIM_PLUGIN_PATH"] = (
        str(PLUGIN_STAGE)
        if not old_plugin_path
        else f"{PLUGIN_STAGE}{os.pathsep}{old_plugin_path}"
    )

import mim
import mim.plug.regex as regex


FUTAMURA_SOURCE = Path(__file__).with_name("futamura.mim")
LOWER_REGEX_SOURCE = Path(__file__).with_name("lower_regex.mim")
POPL25_REGEX_BENCH = Path("/workspaces/compile/popl25/mimir_regex_benchmark")
PAPER_EMAIL_EXTRACT_RE = re.compile(rb"[a-zA-Z0-9_.\-]+@[a-zA-Z0-9_.\-]+")
PAPER_EMAIL_MATCH_RE = re.compile(
    rb"^[a-zA-Z0-9](?:[a-zA-Z0-9]*[._\-]+[a-zA-Z0-9])*[a-zA-Z0-9]*"
    rb"@[a-zA-Z0-9](?:[a-zA-Z0-9]*[_\-]+[a-zA-Z0-9])*[a-zA-Z0-9]*"
    rb"\.(?:(?:[a-zA-Z0-9]*[_\-]+[a-zA-Z0-9])*[a-zA-Z0-9]+\.)*"
    rb"[a-zA-Z][a-zA-Z]+$"
)


def repeat_to(seed: bytes, length: int) -> bytes:
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


@dataclass(frozen=True)
class Pattern:
    name: str
    rust: str
    build_python: Callable[[regex.RegBuilder], regex.MimRegex]
    examples: tuple[tuple[bytes, bool], ...]
    workloads: tuple[tuple[str, bytes], ...]


def a_plus_b(builder: regex.RegBuilder) -> regex.MimRegex:
    return builder.lit("a")["+"] + builder.lit("b")


def literal_abc(builder: regex.RegBuilder) -> regex.MimRegex:
    return builder.lit("abc")


def abc_plus(builder: regex.RegBuilder) -> regex.MimRegex:
    return builder.lit("abc")["+"]


def ascii_word_plus(builder: regex.RegBuilder) -> regex.MimRegex:
    return (builder.alnum() | builder.lit("_"))["+"]


def digit_plus(builder: regex.RegBuilder) -> regex.MimRegex:
    return builder.range("0", "9")["+"]


def identifier(builder: regex.RegBuilder) -> regex.MimRegex:
    first = builder.alpha() | builder.lit("_")
    rest = builder.alnum() | builder.lit("_")
    return first + rest["*"]


def color_optional(builder: regex.RegBuilder) -> regex.MimRegex:
    return builder.lit("colo") + builder.lit("u")["?"] + builder.lit("r")


def fowler_basic26(builder: regex.RegBuilder) -> regex.MimRegex:
    return (builder.lit("ab") | builder.lit("a")) + (builder.lit("bc") | builder.lit("c"))


def keyword_alt(builder: regex.RegBuilder) -> regex.MimRegex:
    return builder.lit("cat") | builder.lit("car") | builder.lit("dog")


def alt(first: regex.MimRegex, *rest: regex.MimRegex) -> regex.MimRegex:
    result = first
    for item in rest:
        result = result | item
    return result


def paper_email(builder: regex.RegBuilder) -> regex.MimRegex:
    letter_or_digit = builder.alnum()
    letter = builder.alpha()
    user_sep = alt(builder.lit("."), builder.lit("_"), builder.lit("-"))
    domain_sep = builder.lit("_") | builder.lit("-")

    user = (
        letter_or_digit
        + (letter_or_digit["*"] + user_sep["+"] + letter_or_digit)["*"]
        + letter_or_digit["*"]
    )
    domain_label_tail = (letter_or_digit["*"] + domain_sep["+"] + letter_or_digit)["*"]
    domain = (
        letter_or_digit
        + domain_label_tail
        + letter_or_digit["*"]
        + builder.lit(".")
        + (domain_label_tail + letter_or_digit["+"] + builder.lit("."))["*"]
        + letter
        + letter["+"]
    )
    return user + builder.lit("@") + domain


PATTERNS = (
    Pattern(
        name="literal_abc",
        rust="abc",
        build_python=literal_abc,
        examples=(
            (b"abc", True),
            (b"", False),
            (b"ab", False),
            (b"abcd", False),
            (b"xabc", False),
        ),
        workloads=(
            ("short-match", b"abc"),
            ("short-reject", b"abX"),
            ("long-match", b"abc"),
            ("late-reject", b"abX"),
        ),
    ),
    Pattern(
        name="a_plus_b",
        rust="a+b",
        build_python=a_plus_b,
        examples=((b"ab", True), (b"aaab", True), (b"a", False), (b"abc", False), (b"xb", False)),
        workloads=(
            ("short-match", b"aaab"),
            ("short-reject", b"aaaX"),
            ("medium-match", b"a" * 255 + b"b"),
            ("medium-reject", b"a" * 255 + b"X"),
            ("long-match", b"a" * 4095 + b"b"),
            ("late-reject", b"a" * 4095 + b"X"),
            ("huge-match", b"a" * 65535 + b"b"),
            ("huge-reject", b"a" * 65535 + b"X"),
        ),
    ),
    Pattern(
        name="abc_plus",
        rust="(?:abc)+",
        build_python=abc_plus,
        examples=(
            (b"abc", True),
            (b"abcabc", True),
            (b"", False),
            (b"ab", False),
            (b"abcd", False),
            (b"xyz", False),
        ),
        workloads=(
            ("short-match", b"abcabc"),
            ("short-reject", b"abcabX"),
            ("medium-match", b"abc" * 85),
            ("medium-reject", b"abc" * 84 + b"abX"),
            ("long-match", b"abc" * 1365),
            ("late-reject", b"abc" * 1364 + b"abX"),
            ("huge-match", b"abc" * 21845),
            ("huge-reject", b"abc" * 21844 + b"abX"),
        ),
    ),
    Pattern(
        name="ascii_word_plus",
        rust=r"[A-Za-z0-9_]+",
        build_python=ascii_word_plus,
        examples=(
            (b"abc_09XYZ", True),
            (b"Z", True),
            (b"", False),
            (b"abc-def", False),
            (b"abc!", False),
        ),
        workloads=(
            ("short-match", b"abc_09XYZ"),
            ("short-reject", b"abc-09"),
            ("medium-match", repeat_to(b"abc_09XYZ", 256)),
            ("medium-reject", repeat_to(b"abc_09XYZ", 255) + b"!"),
            ("long-match", (b"abc_09XYZ" * 455)[:4095]),
            ("late-reject", (b"abc_09XYZ" * 455)[:4094] + b"!"),
            ("huge-match", repeat_to(b"abc_09XYZ", 65536)),
            ("huge-reject", repeat_to(b"abc_09XYZ", 65535) + b"!"),
        ),
    ),
    Pattern(
        name="digit_plus",
        rust=r"[0-9]+",
        build_python=digit_plus,
        examples=(
            (b"123456", True),
            (b"0", True),
            (b"", False),
            (b"123a", False),
            (b"a123", False),
        ),
        workloads=(
            ("short-match", b"123456"),
            ("short-reject", b"12345X"),
            ("medium-match", repeat_to(b"0123456789", 256)),
            ("medium-reject", repeat_to(b"0123456789", 255) + b"X"),
            ("long-match", repeat_to(b"0123456789", 4096)),
            ("late-reject", repeat_to(b"0123456789", 4095) + b"X"),
            ("huge-match", repeat_to(b"0123456789", 65536)),
            ("huge-reject", repeat_to(b"0123456789", 65535) + b"X"),
        ),
    ),
    Pattern(
        name="identifier",
        rust=r"[A-Za-z_][A-Za-z0-9_]*",
        build_python=identifier,
        examples=(
            (b"abc_09XYZ", True),
            (b"_", True),
            (b"a", True),
            (b"9abc", False),
            (b"abc-def", False),
        ),
        workloads=(
            ("short-match", b"_abc09"),
            ("short-reject", b"9abc09"),
            ("medium-match", b"a" + repeat_to(b"bc_09XYZ", 255)),
            ("medium-reject", b"a" + repeat_to(b"bc_09XYZ", 254) + b"!"),
            ("long-match", b"a" + repeat_to(b"bc_09XYZ", 4095)),
            ("late-reject", b"a" + repeat_to(b"bc_09XYZ", 4094) + b"!"),
            ("huge-match", b"a" + repeat_to(b"bc_09XYZ", 65535)),
            ("huge-reject", b"a" + repeat_to(b"bc_09XYZ", 65534) + b"!"),
        ),
    ),
    Pattern(
        name="color_optional",
        rust=r"colou?r",
        build_python=color_optional,
        examples=(
            (b"color", True),
            (b"colour", True),
            (b"colouur", False),
            (b"colr", False),
            (b"xcolor", False),
        ),
        workloads=(
            ("short-match", b"colour"),
            ("short-reject", b"colouX"),
            ("long-match", b"colour"),
            ("late-reject", b"colouX"),
        ),
    ),
    Pattern(
        name="keyword_alt",
        rust=r"(?:cat|car|dog)",
        build_python=keyword_alt,
        examples=(
            (b"cat", True),
            (b"car", True),
            (b"dog", True),
            (b"cow", False),
            (b"cater", False),
        ),
        workloads=(
            ("short-match", b"cat"),
            ("short-reject", b"cow"),
            ("long-match", b"dog"),
            ("late-reject", b"caz"),
        ),
    ),
    Pattern(
        name="fowler_basic26",
        rust=r"(?:ab|a)(?:bc|c)",
        build_python=fowler_basic26,
        examples=(
            (b"abc", True),
            (b"ac", True),
            (b"abbc", True),
            (b"", False),
            (b"ab", False),
            (b"abb", False),
            (b"abcc", False),
        ),
        workloads=(
            ("short-match", b"abbc"),
            ("short-reject", b"abbX"),
            ("long-match", b"abbc"),
            ("late-reject", b"abbX"),
        ),
    ),
    Pattern(
        name="paper_email",
        rust=(
            r"[a-zA-Z0-9](?:[a-zA-Z0-9]*[._\-]+[a-zA-Z0-9])*[a-zA-Z0-9]*"
            r"@[a-zA-Z0-9](?:[a-zA-Z0-9]*[_\-]+[a-zA-Z0-9])*[a-zA-Z0-9]*"
            r"\.(?:(?:[a-zA-Z0-9]*[_\-]+[a-zA-Z0-9])*[a-zA-Z0-9]+\.)*"
            r"[a-zA-Z][a-zA-Z]+"
        ),
        build_python=paper_email,
        examples=(
            (b"alice@example.com", True),
            (b"a.b-c@example_domain.com", True),
            (b"bob.smith-42@sub-domain.example.co", True),
            (b".alice@example.com", False),
            (b"alice@example", False),
            (b"alice@example.c", False),
        ),
        workloads=(
            ("paper-valid", b"bob.smith-42@sub-domain.example.co"),
            ("paper-invalid", b"alice@example.c"),
        ),
    ),
)


@dataclass
class BatchMatcher:
    name: str
    run: Callable[[bytes, int], int]
    run_corpus: Callable[[list[bytes], list[bool], int], int] | None = None


def check_tools(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"missing required tools: {', '.join(missing)}")


def run(command: list[str], cwd: Path, *, quiet: bool = False, env: dict[str, str] | None = None) -> None:
    kwargs = {"cwd": cwd, "check": True}
    if env is not None:
        kwargs["env"] = env
    if quiet:
        kwargs |= {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "text": True}
    subprocess.run(command, **kwargs)


def write_batch_shim(path: Path, functions: dict[str, str]) -> None:
    declarations = []
    definitions = []
    for exported, matcher in functions.items():
        declarations.append(f"extern bool {matcher}(const char *input);")
        definitions.append(
            f"""
uint64_t {exported}(const char *input, uint64_t iterations) {{
    bool (*volatile call)(const char *) = {matcher};
    uint64_t matches = 0;
    for (uint64_t i = 0; i < iterations; ++i)
        matches += call(input);
    return matches;
}}

uint64_t {exported}_corpus(const char *const *inputs, const uint8_t *expected, uint64_t count, uint64_t iterations) {{
    bool (*volatile call)(const char *) = {matcher};
    uint64_t mismatches = 0;
    for (uint64_t it = 0; it < iterations; ++it) {{
        for (uint64_t i = 0; i < count; ++i)
            mismatches += call(inputs[i]) != (bool)expected[i];
    }}
    return mismatches;
}}
"""
        )
    path.write_text(
        "#include <stdbool.h>\n#include <stdint.h>\n"
        + "\n".join(declarations)
        + "\n"
        + "\n".join(definitions)
    )


def load_c_batch_library(path: Path, names: list[str], engine: str) -> dict[str, BatchMatcher]:
    library = ctypes.CDLL(path)
    result = {}
    for name in names:
        symbol = f"bench_{name}"
        function = getattr(library, symbol)
        function.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        function.restype = ctypes.c_uint64
        corpus_function = getattr(library, f"{symbol}_corpus")
        corpus_function.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        corpus_function.restype = ctypes.c_uint64

        def run_corpus(
            inputs: list[bytes],
            expected: list[bool],
            iterations: int,
            f=corpus_function,
        ) -> int:
            input_array = (ctypes.c_char_p * len(inputs))(*inputs)
            expected_array = (ctypes.c_uint8 * len(expected))(*(1 if x else 0 for x in expected))
            return int(f(input_array, expected_array, len(inputs), iterations))

        result[name] = BatchMatcher(engine, lambda data, n, f=function: int(f(data, n)), run_corpus)
    return result


def build_python_matchers(work: Path) -> tuple[dict[str, BatchMatcher], float]:
    matchers = {}
    elapsed = 0.0
    old_cwd = Path.cwd()
    try:
        for pattern in PATTERNS:
            pattern_dir = work / f"python-{pattern.name}"
            pattern_dir.mkdir()
            os.chdir(pattern_dir)
            start = time.perf_counter()
            driver = mim.Driver()
            builder = regex.RegBuilder(driver, f"python_{pattern.name}", mim.Level.Error)
            expression = pattern.build_python(builder)
            expression.jit()
            elapsed += time.perf_counter() - start

            shim = pattern_dir / "batch.c"
            write_batch_shim(shim, {f"bench_{pattern.name}": "match_func"})
            output = pattern_dir / "batch.so"
            run(
                [
                    "clang",
                    f"python_{pattern.name}.ll",
                    str(shim),
                    "-O3",
                    "-shared",
                    "-fPIC",
                    "-Wno-override-module",
                    "-o",
                    str(output),
                ],
                pattern_dir,
            )
            matchers.update(load_c_batch_library(output, [pattern.name], "python-lower-regex"))
    finally:
        os.chdir(old_cwd)
    return matchers, elapsed


def build_graal_style_matchers(work: Path, mim_binary: Path) -> tuple[dict[str, BatchMatcher], float]:
    output_dir = work / "graal-style"
    output_dir.mkdir()
    start = time.perf_counter()
    run([str(mim_binary), "-p", "opt", str(FUTAMURA_SOURCE), "-o", "-", "-p", "ll"], output_dir, quiet=True)
    shim = output_dir / "batch.c"
    write_batch_shim(shim, {f"bench_{p.name}": f"match_{p.name}" for p in PATTERNS})
    output = output_dir / "batch.so"
    run(
        [
            "clang",
            FUTAMURA_SOURCE.with_suffix(".ll").name,
            str(shim),
            "-O3",
            "-shared",
            "-fPIC",
            "-Wno-override-module",
            "-o",
            str(output),
        ],
        output_dir,
    )
    elapsed = time.perf_counter() - start
    return load_c_batch_library(output, [p.name for p in PATTERNS], "graal-style-pe"), elapsed


def build_native_lower_regex_matchers(work: Path, mim_binary: Path) -> tuple[dict[str, BatchMatcher], float]:
    output_dir = work / "native-lower-regex"
    output_dir.mkdir()
    start = time.perf_counter()
    run([str(mim_binary), "-p", "opt", str(LOWER_REGEX_SOURCE), "-o", "-", "-p", "ll"], output_dir, quiet=True)
    shim = output_dir / "batch.c"
    write_batch_shim(shim, {f"bench_{p.name}": f"match_{p.name}" for p in PATTERNS})
    output = output_dir / "batch.so"
    run(
        [
            "clang",
            "lower_regex.ll",
            str(shim),
            "-O3",
            "-shared",
            "-fPIC",
            "-Wno-override-module",
            "-o",
            str(output),
        ],
        output_dir,
    )
    elapsed = time.perf_counter() - start
    return load_c_batch_library(output, [p.name for p in PATTERNS], "native-lower-regex"), elapsed


def rust_source() -> str:
    declarations = []
    for pattern in PATTERNS:
        upper = pattern.name.upper()
        declarations.append(
            rf"""
static {upper}: OnceLock<Regex> = OnceLock::new();

#[inline(never)]
fn match_{pattern.name}(haystack: &[u8]) -> bool {{
    {upper}.get_or_init(|| Regex::new(r"\A(?:{pattern.rust})\z").unwrap()).is_match(haystack)
}}

#[no_mangle]
pub unsafe extern "C" fn bench_{pattern.name}(ptr: *const u8, len: usize, iterations: u64) -> u64 {{
    let haystack = std::slice::from_raw_parts(ptr, len);
    let mut matches = 0;
    for _ in 0..iterations {{
        matches += match_{pattern.name}(std::hint::black_box(haystack)) as u64;
    }}
    std::hint::black_box(matches)
}}

#[no_mangle]
pub unsafe extern "C" fn bench_{pattern.name}_corpus(
    inputs: *const *const std::ffi::c_char,
    expected: *const u8,
    count: usize,
    iterations: u64,
) -> u64 {{
    let mut mismatches = 0;
    for _ in 0..iterations {{
        for i in 0..count {{
            let input = *inputs.add(i);
            let haystack = std::ffi::CStr::from_ptr(input).to_bytes();
            let wanted = *expected.add(i) != 0;
            mismatches += (match_{pattern.name}(std::hint::black_box(haystack)) != wanted) as u64;
        }}
    }}
    std::hint::black_box(mismatches)
}}
"""
        )
    return "use regex::bytes::Regex;\nuse std::sync::OnceLock;\n" + "\n".join(declarations)


def build_rust_matchers(
    work: Path,
    rust_regex_dir: Path,
    engine: str = "rust-regex-bytes",
    rustflags: str | None = None,
) -> tuple[dict[str, BatchMatcher], float]:
    output_dir = work / engine
    (output_dir / "src").mkdir(parents=True)
    manifest = f"""[package]
name = "mimir-regex-compare"
version = "0.0.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
regex = {{ path = {str(rust_regex_dir)!r} }}
"""
    (output_dir / "Cargo.toml").write_text(manifest)
    (output_dir / "src" / "lib.rs").write_text(rust_source())
    start = time.perf_counter()
    env = os.environ.copy()
    if rustflags is not None:
        env["RUSTFLAGS"] = rustflags
    run(["cargo", "build", "--release", "--offline"], output_dir, env=env)
    elapsed = time.perf_counter() - start
    library_path = output_dir / "target" / "release" / "libmimir_regex_compare.so"
    library = ctypes.CDLL(library_path)
    result = {}
    for pattern in PATTERNS:
        function = getattr(library, f"bench_{pattern.name}")
        function.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
        function.restype = ctypes.c_uint64
        corpus_function = getattr(library, f"bench_{pattern.name}_corpus")
        corpus_function.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        corpus_function.restype = ctypes.c_uint64

        def run_corpus(
            inputs: list[bytes],
            expected: list[bool],
            iterations: int,
            f=corpus_function,
        ) -> int:
            input_array = (ctypes.c_char_p * len(inputs))(*inputs)
            expected_array = (ctypes.c_uint8 * len(expected))(*(1 if x else 0 for x in expected))
            return int(f(input_array, expected_array, len(inputs), iterations))

        result[pattern.name] = BatchMatcher(
            engine,
            lambda data, n, f=function: int(f(data, len(data), n)),
            run_corpus,
        )
    return result, elapsed


def verify(all_matchers: dict[str, dict[str, BatchMatcher]]) -> None:
    for pattern in PATTERNS:
        for data, expected in pattern.examples:
            for engine, matchers in all_matchers.items():
                if pattern.name not in matchers:
                    continue
                actual = matchers[pattern.name].run(data, 1) == 1
                if actual != expected:
                    raise AssertionError(
                        f"{engine}/{pattern.name}: {data!r}: expected {expected}, got {actual}"
                    )


def benchmark(
    all_matchers: dict[str, dict[str, BatchMatcher]], iterations: int, samples: int
) -> list[tuple[str, str, str, int, float, float]]:
    rows = []
    for pattern in PATTERNS:
        for workload, data in pattern.workloads:
            expected = dict(pattern.examples).get(data)
            for engine, matchers in all_matchers.items():
                if pattern.name not in matchers:
                    continue
                matcher = matchers[pattern.name]
                matcher.run(data, min(iterations, 1000))
                timings = []
                for _ in range(samples):
                    start = time.perf_counter_ns()
                    count = matcher.run(data, iterations)
                    elapsed = time.perf_counter_ns() - start
                    if expected is not None:
                        wanted = iterations if expected else 0
                        if count != wanted:
                            raise AssertionError(f"unstable result for {engine}/{pattern.name}/{workload}")
                    timings.append(elapsed)
                median = statistics.median(timings)
                ns_per_match = median / iterations
                mib_per_second = (len(data) * iterations) / median * (1e9 / (1024 * 1024))
                rows.append((pattern.name, workload, engine, len(data), ns_per_match, mib_per_second))
    return rows


def load_paper_email_corpus(path: Path) -> tuple[list[bytes], list[bool]]:
    if path.name == "fradulent_emails.txt":
        extracted = set(PAPER_EMAIL_EXTRACT_RE.findall(path.read_bytes()))
        inputs = sorted(extracted)
        expected = [PAPER_EMAIL_MATCH_RE.match(item) is not None for item in inputs]
        return inputs, expected

    inputs = []
    expected = []
    for line in path.read_bytes().splitlines():
        if not line:
            continue
        data, _, label = line.rpartition(b",")
        if not data:
            raise RuntimeError(f"invalid annotated corpus line: {line!r}")
        inputs.append(data)
        expected.append(label.lower() in {b"1", b"true"})
    return inputs, expected


def benchmark_corpus(
    all_matchers: dict[str, dict[str, BatchMatcher]],
    inputs: list[bytes],
    expected: list[bool],
    iterations: int,
    samples: int,
) -> list[tuple[str, str, str, int, float, float]]:
    pattern_name = "paper_email"
    total_bytes = sum(len(item) for item in inputs)
    rows = []
    for engine, matchers in all_matchers.items():
        matcher = matchers.get(pattern_name)
        if matcher is None or matcher.run_corpus is None:
            continue
        mismatches = matcher.run_corpus(inputs, expected, 1)
        if mismatches:
            raise AssertionError(f"{engine}/{pattern_name}/paper-corpus: {mismatches} mismatches")
        timings = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            mismatches = matcher.run_corpus(inputs, expected, iterations)
            elapsed = time.perf_counter_ns() - start
            if mismatches:
                raise AssertionError(f"{engine}/{pattern_name}/paper-corpus: {mismatches} mismatches")
            timings.append(elapsed)
        median = statistics.median(timings)
        items = len(inputs) * iterations
        ns_per_item = median / items
        mib_per_second = (total_bytes * iterations) / median * (1e9 / (1024 * 1024))
        rows.append((pattern_name, f"paper-corpus-{len(inputs)}", engine, total_bytes, ns_per_item, mib_per_second))
    return rows


def print_results(rows: list[tuple[str, str, str, int, float, float]], compile_times: dict[str, float]) -> None:
    print("\nBuild/setup time (not directly comparable):")
    for engine, elapsed in compile_times.items():
        print(f"  {engine:20s} {elapsed:9.3f} s")
    print("\n| Pattern | Workload | Bytes | Engine | ns/match | MiB/s |")
    print("|---|---|---:|---|---:|---:|")
    for pattern, workload, engine, size, latency, throughput in rows:
        print(f"| {pattern} | {workload} | {size} | {engine} | {latency:.1f} | {throughput:.1f} |")


def parse_args() -> argparse.Namespace:
    default_rust = Path(os.environ.get("RUST_REGEX_DIR", "/workspaces/rust/regex"))
    default_paper_corpus = POPL25_REGEX_BENCH / "fradulent_emails.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-regex-dir", type=Path, default=default_rust)
    parser.add_argument("--mim", type=Path, default=ROOT / "build" / "bin" / "mim")
    parser.add_argument("--iterations", type=int, default=3_000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--paper-email-corpus", type=Path, default=default_paper_corpus)
    parser.add_argument("--no-paper-email-corpus", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_tools("clang", "cargo")
    if not args.rust_regex_dir.joinpath("Cargo.toml").is_file():
        raise RuntimeError(f"not a rust-lang/regex checkout: {args.rust_regex_dir}")
    if not args.mim.is_file():
        raise RuntimeError(f"MimIR compiler not found: {args.mim}")

    with tempfile.TemporaryDirectory(prefix="mimir-regex-compare-") as tmp:
        work = Path(tmp)
        python_matchers, python_time = build_python_matchers(work)
        native_matchers, native_time = build_native_lower_regex_matchers(work, args.mim)
        graal_matchers, graal_time = build_graal_style_matchers(work, args.mim)
        rust_matchers, rust_time = build_rust_matchers(work, args.rust_regex_dir.resolve())
        rust_native_matchers, rust_native_time = build_rust_matchers(
            work,
            args.rust_regex_dir.resolve(),
            engine="rust-regex-bytes-native",
            rustflags="-C target-cpu=native",
        )
        all_matchers = {
            "python-lower-regex": python_matchers,
            "native-lower-regex": native_matchers,
            "rust-regex-bytes": rust_matchers,
            "rust-regex-bytes-native": rust_native_matchers,
            "graal-style-pe": graal_matchers,
        }
        verify(all_matchers)
        print("Cross-engine correctness: PASS")
        if args.check_only:
            return
        rows = benchmark(all_matchers, args.iterations, args.samples)
        if not args.no_paper_email_corpus:
            if not args.paper_email_corpus.is_file():
                raise RuntimeError(f"paper email corpus not found: {args.paper_email_corpus}")
            corpus_inputs, corpus_expected = load_paper_email_corpus(args.paper_email_corpus)
            rows.extend(benchmark_corpus(all_matchers, corpus_inputs, corpus_expected, 1, args.samples))
        print_results(
            rows,
            {
                "python-lower-regex": python_time,
                "native-lower-regex": native_time,
                "rust-regex-bytes": rust_time,
                "rust-regex-bytes-native": rust_native_time,
                "graal-style-pe": graal_time,
            },
        )


if __name__ == "__main__":
    main()
