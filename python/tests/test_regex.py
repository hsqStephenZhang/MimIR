"""
Transpile Python's regex IR (sre_parse) into MimIR's regex dialect and verify
the resulting IR nodes.

Opcode mapping:
  LITERAL c              -> %regex.lit c          (lam: range(c,c))
  ANY                    -> %regex.any
  MAX_REPEAT(0,MAX,p)    -> %regex.quant.star(p)
  MAX_REPEAT(1,MAX,p)    -> %regex.quant.plus(p)
  MAX_REPEAT(0,1,p)      -> %regex.quant.optional(p)
  BRANCH(branches)       -> %regex.disj(branches)
  IN [RANGE lo hi]       -> %regex.range(lo, hi)
  IN [LITERAL c]         -> %regex.range(c, c)
  IN [CATEGORY cat]      -> %regex.cls.{d,D,w,W,s,S}
  IN [NEGATE, ...]       -> %regex.not_(...)
  sequence               -> %regex.conj(parts)    (1-element: no wrap)

Known limitations:
- the re._parser module is private, so the type checker may not be happy

"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import re._parser as _sre_parse  # type: ignore
from sre_constants import (  # type: ignore[import]
    ANY,
    BRANCH,
    CATEGORY,
    CATEGORY_DIGIT,
    CATEGORY_NOT_DIGIT,
    CATEGORY_NOT_SPACE,
    CATEGORY_NOT_WORD,
    CATEGORY_SPACE,
    CATEGORY_WORD,
    IN,
    LITERAL,
    MAX_REPEAT,
    MAXREPEAT,
    MIN_REPEAT,
    NEGATE,
    RANGE,
    SUBPATTERN,
)
from typing import TYPE_CHECKING, Any

import pytest

import mim
from mim import Regex

if TYPE_CHECKING:
    from mim._mim_core import Def, Driver, World


@dataclass(frozen=True)
class RegexBenchCase:
    name: str
    pattern: str
    sample: bytes


@pytest.fixture()
def regex_world() -> tuple[Driver, World]:
    driver = mim.make_driver("compile", "mem", "core", "regex")
    return driver, driver.world()


# ---------------------------------------------------------------------------
# Transpiler
# ---------------------------------------------------------------------------

_BENCH_CASES = [
    RegexBenchCase(
        name="simple_literal",
        pattern="a",
        sample=(b"a" * 4096) + b"z",
    ),
    RegexBenchCase(
        name="simple_class_plus",
        pattern=r"[a-z]+",
        sample=(b"hello" * 1024) + b"123",
    ),
    RegexBenchCase(
        name="complex_identifier",
        pattern=r"[a-zA-Z_][a-zA-Z_0-9]*",
        sample=(b"identifier_123_" * 512) + b"!",
    ),
    RegexBenchCase(
        name="complex_emailish",
        pattern=r"[a-z]+@[a-z]+",
        sample=(b"helloworld@example" * 256) + b"!",
    ),
    RegexBenchCase(
        name="complex_alternation",
        pattern=r"(ab|cd)+",
        sample=(b"abcd" * 1024) + b"x",
    ),
]


_CATEGORY_AXIOMS = {
    CATEGORY_DIGIT: Regex.cls.d,
    CATEGORY_NOT_DIGIT: Regex.cls.D,
    CATEGORY_WORD: Regex.cls.w,
    CATEGORY_NOT_WORD: Regex.cls.W,
    CATEGORY_SPACE: Regex.cls.s,
    CATEGORY_NOT_SPACE: Regex.cls.S,
}


class RegexTranspiler:
    """Translates a Python regex pattern into a MimIR RE def."""

    def __init__(self, world: World) -> None:
        self.world = world

    def transpile(self, pattern: str) -> Def:
        ops = list(_sre_parse.parse(pattern))
        return self._seq(ops)

    # -- internal ------------------------------------------------------------

    def _seq(self, ops: list) -> Def:
        """Translate a sequence; wraps in %regex.conj when len > 1."""
        parts = [self._op(op, arg) for op, arg in ops]
        if not parts:
            return Regex.empty(self.world)
        if len(parts) == 1:
            return parts[0]
        # %regex.conj: {n: Nat} → «n; RE» → RE  — implicit n inferred from tuple arity
        return Regex.conj(self.world, parts, implicit=True)

    def _op(self, op: int, arg: Any) -> Def:
        w = self.world

        if op == LITERAL:
            return Regex.lit(w, w.lit_i8(arg))  # type: ignore[arg-type]

        if op == ANY:
            return Regex.any(w)

        if op == SUBPATTERN:
            # (group_id, add_flags, del_flags, pattern)
            return self._seq(list(arg[-1]))  # type: ignore[index]

        if op == MAX_REPEAT or op == MIN_REPEAT:
            min_, max_, pattern = arg  # type: ignore[misc]
            inner = self._seq(list(pattern))
            if min_ == 0 and max_ == MAXREPEAT:
                return Regex.quant.star(w, inner)
            if min_ == 1 and max_ == MAXREPEAT:
                return Regex.quant.plus(w, inner)
            if min_ == 0 and max_ == 1:
                return Regex.quant.optional(w, inner)
            raise NotImplementedError(f"Bounded repeat {{{min_},{max_}}} not supported")

        if op == BRANCH:
            _, branches = arg  # type: ignore[misc]
            parts = [self._seq(list(b)) for b in branches]
            # %regex.disj: {n: Nat} → «n; RE» → RE  — implicit n
            return Regex.disj(w, parts, implicit=True)

        if op == IN:
            return self._in(arg)  # type: ignore[arg-type]

        raise NotImplementedError(f"Unsupported sre opcode: {op!r}")

    def _in(self, items: list) -> Def:
        """Translate a character-class or single-char alternation (IN node)."""
        w = self.world
        negate = False
        parts: list[Def] = []

        for item_op, item_arg in items:
            if item_op == NEGATE:
                negate = True
                continue
            if item_op == LITERAL:
                parts.append(
                    Regex.range(w, w.tuple([w.lit_i8(item_arg), w.lit_i8(item_arg)]))
                )
            elif item_op == RANGE:
                lo, hi = item_arg
                parts.append(Regex.range(w, w.tuple([w.lit_i8(lo), w.lit_i8(hi)])))
            elif item_op == CATEGORY:
                axiom = _CATEGORY_AXIOMS.get(item_arg)
                if axiom is None:
                    raise NotImplementedError(f"Unsupported sre category: {item_arg!r}")
                parts.append(axiom(w))
            else:
                raise NotImplementedError(f"Unsupported IN item opcode: {item_op!r}")

        assert parts, "IN node produced no sub-expressions"
        result = parts[0] if len(parts) == 1 else Regex.disj(w, parts, implicit=True)
        return Regex.not_(w, result) if negate else result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ir(world: World, pattern: str) -> str:
    """Build and return the MimIR string for the transpiled pattern."""
    return RegexTranspiler(world).transpile(pattern).to_string()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiteral:
    def test_single_char_returns_def(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert RegexTranspiler(world).transpile("a") is not None

    def test_char_code_present_in_ir(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, "a")
        # 'a' == 97 == 0x61; MimIR may use hex or decimal
        assert "61" in s or "97" in s, f"char code missing from IR: {s}"

    def test_distinct_chars_produce_distinct_ir(self, regex_world: tuple) -> None:
        _, world = regex_world
        t = RegexTranspiler(world)
        assert t.transpile("a").to_string() != t.transpile("b").to_string()

    def test_capturing_group_is_transparent(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert _ir(world, "(a)") == _ir(world, "a")

    def test_two_literals_wrapped_in_conj(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "conj" in _ir(world, "ab")

    def test_single_literal_no_conj(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "conj" not in _ir(world, "a")

    def test_three_literals_has_conj(self, regex_world: tuple) -> None:
        # MimIR normalizes n-ary conj into binary trees, so count >= 1
        _, world = regex_world
        assert "conj" in _ir(world, "abc")


class TestQuantifiers:
    def test_star(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "star" in _ir(world, "a*")

    def test_plus(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "plus" in _ir(world, "a+")

    def test_optional(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "optional" in _ir(world, "a?")

    def test_star_of_sequence(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, "(ab)*")
        assert "star" in s
        assert "conj" in s

    def test_bounded_repeat_raises(self, regex_world: tuple) -> None:
        _, world = regex_world
        with pytest.raises(NotImplementedError, match="Bounded repeat"):
            RegexTranspiler(world).transpile("a{2,4}")

    def test_lazy_star_maps_to_star(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "star" in _ir(world, "a*?")

    def test_lazy_plus_maps_to_plus(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "plus" in _ir(world, "a+?")

    def test_lazy_optional_maps_to_optional(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "optional" in _ir(world, "a??")


class TestAlternation:
    def test_two_chars_alt(self, regex_world: tuple) -> None:
        # sre folds `a|b` into IN; both chars become ranges or a disj
        _, world = regex_world
        s = _ir(world, "a|b")
        assert "range" in s or "disj" in s

    def test_sequence_alt_uses_disj(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "disj" in _ir(world, "ab|cd")

    def test_three_way_alt(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "disj" in _ir(world, "ab|cd|ef")


class TestCharacterClasses:
    def test_range(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "range" in _ir(world, "[a-z]")

    def test_negated_range(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, "[^a-z]")
        assert "not_" in s
        assert "range" in s

    def test_multi_range_class_uses_disj(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, "[a-zA-Z]")
        assert "disj" in s
        assert "range" in s

    def test_digit_shorthand(self, regex_world: tuple) -> None:
        # \d normalizes to range('0','9')
        _, world = regex_world
        assert "range" in _ir(world, r"\d")

    def test_word_shorthand(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, r"\w")
        assert "range" in s or "disj" in s

    def test_not_word_shorthand(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, r"\W")
        assert "not_" in s
        assert "range" in s or "disj" in s

    def test_not_space_shorthand(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, r"\S")
        assert "not_" in s
        assert "range" in s or "disj" in s

    def test_not_digit_shorthand(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, r"\D")
        assert "not_" in s
        assert "range" in s


class TestAny:
    def test_dot_produces_any(self, regex_world: tuple) -> None:
        _, world = regex_world
        assert "any" in _ir(world, ".")


class TestComplex:
    def test_identifier_pattern(self, regex_world: tuple) -> None:
        # [a-zA-Z_][a-zA-Z_0-9]*
        _, world = regex_world
        s = _ir(world, r"[a-zA-Z_][a-zA-Z_0-9]*")
        assert "star" in s
        assert "range" in s

    def test_digits_plus_letters(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, r"[a-z]+@[a-z]+")
        assert "plus" in s
        assert "conj" in s

    def test_optional_group(self, regex_world: tuple) -> None:
        _, world = regex_world
        s = _ir(world, r"colou?r")
        assert "optional" in s
        assert "conj" in s


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

# match_func signature after CPS optimisation:
#   bool match_func(const char* str)
# — mem is zero-sized and optimised away; the return continuation becomes the
#   native return path.


def _build_matcher(world: World, regex_ir: Def) -> None:
    """
    Wire `regex_ir` into an externally-visible CPS continuation:

        match_func : cn [%mem.M, %mem.Ptr0(«⊤; I8»), cn[%mem.M, Bool]]

    After optimize() + LLVM lowering this becomes bool(const char*).
    """
    n = world.top_nat()
    mem_t = world.call(r"%mem.M", world.lit_nat_0())
    str_t = world.call(r"%mem.Ptr0", [world.arr(n, world.type_i8())])
    ret_t = world.cn([mem_t, world.type_bool()])

    fn = world.mut_con([mem_t, str_t, ret_t]).set("match_func")
    fn.externalize()

    mem = fn.var().proj(0)
    str_ptr = fn.var().proj(1)
    ret = fn.var().proj(2)

    # Apply regex: implicit n is inferred from str_ptr's array type
    result = world.implicit_app(
        regex_ir, [mem, str_ptr, world.lit(world.type_idx(n), 0)]
    )
    fn.app(False, ret, [result.proj(0), result.proj(1)])


def _compile_pattern(pattern: str, tmp: Path) -> Callable[[bytes], bool]:
    """Transpile, compile, and return a callable bool(bytes) matcher."""
    from mim._mim_core import AST, Parser

    driver = mim.make_driver("compile", "mem", "core", "regex", "opt")
    world = driver.world()

    # Parser.plugin() parses the .mim file, making lam/let defs (including
    # _default_compile from opt.mim) available. load_plugins() only registers
    # C++ normalizers/stages — it does NOT parse .mim definitions.
    ast = AST(world)
    p = Parser(ast)
    for plugin_name in ("compile", "mem", "core", "regex", "opt"):
        p.plugin(plugin_name)

    regex_ir = RegexTranspiler(world).transpile(pattern)
    _build_matcher(world, regex_ir)
    world.optimize()

    ll = tmp / "regex.ll"
    so = tmp / "regex.so"
    driver.backend("ll", str(ll), world)
    subprocess.run(
        ["clang", str(ll), "-o", str(so), "-Wno-override-module", "-shared"],
        check=True,
        capture_output=True,
    )

    lib = ctypes.CDLL(str(so))
    lib.match_func.argtypes = [ctypes.c_char_p]
    lib.match_func.restype = ctypes.c_bool
    return lib.match_func


def _python_matcher(pattern: str) -> Callable[[bytes], bool]:
    regex = re.compile(pattern)
    return lambda data: regex.match(data.decode()) is not None


needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang not available"
)


@needs_clang
class TestBenchmarkParity:
    @pytest.mark.parametrize("case", _BENCH_CASES, ids=lambda case: case.name)
    def test_jit_matches_python_re(self, tmp_path: Path, case: RegexBenchCase) -> None:
        case_dir = tmp_path / case.name
        case_dir.mkdir()
        jit = _compile_pattern(case.pattern, case_dir)
        py = _python_matcher(case.pattern)
        assert jit(case.sample) == py(case.sample)


bench_fn = Callable[[Callable[[bytes], bool], bytes], None]


@needs_clang
class TestRegexBenchmark:
    @pytest.mark.benchmark(group="regex-jit")
    @pytest.mark.parametrize("case", _BENCH_CASES, ids=lambda case: case.name)
    def test_jit_benchmark(
        self, benchmark: bench_fn, tmp_path: Path, case: RegexBenchCase
    ) -> None:
        case_dir = tmp_path / f"jit_{case.name}"
        case_dir.mkdir()
        jit = _compile_pattern(case.pattern, case_dir)
        assert jit(case.sample) == _python_matcher(case.pattern)(case.sample)
        benchmark(jit, case.sample)

    @pytest.mark.benchmark(group="regex-python-re")
    @pytest.mark.parametrize("case", _BENCH_CASES, ids=lambda case: case.name)
    def test_python_re_benchmark(
        self, benchmark: bench_fn, case: RegexBenchCase
    ) -> None:
        py = _python_matcher(case.pattern)
        benchmark(py, case.sample)

    def test_single_char_matches(self, tmp_path: Path) -> None:
        f = _compile_pattern("a", tmp_path)
        assert f(b"abc")

    def test_single_char_no_match(self, tmp_path: Path) -> None:
        f = _compile_pattern("a", tmp_path)
        assert not f(b"bcd")


@needs_clang
class TestExecutionQuantifiers:
    def test_star_matches_empty(self, tmp_path: Path) -> None:
        # a* matches the empty prefix of any string
        f = _compile_pattern("a*", tmp_path)
        assert f(b"")
        assert f(b"aaa")
        assert f(b"b")  # empty match at position 0

    def test_plus_requires_one(self, tmp_path: Path) -> None:
        f = _compile_pattern("a+", tmp_path)
        assert f(b"a")
        assert f(b"aaa")
        assert not f(b"bbb")

    def test_lazy_star_matches_empty(self, tmp_path: Path) -> None:
        f = _compile_pattern("a*?", tmp_path)
        assert f(b"")
        assert f(b"aaa")
        assert f(b"b")

    def test_lazy_plus_requires_one(self, tmp_path: Path) -> None:
        f = _compile_pattern("a+?", tmp_path)
        assert f(b"a")
        assert f(b"aaa")
        assert not f(b"bbb")

    def test_range_matches(self, tmp_path: Path) -> None:
        f = _compile_pattern("[a-z]+", tmp_path)
        assert f(b"hello")
        assert not f(b"123")

    def test_digit_shorthand(self, tmp_path: Path) -> None:
        f = _compile_pattern(r"\d+", tmp_path)
        assert f(b"42")
        assert not f(b"abc")
