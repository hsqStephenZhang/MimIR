from __future__ import annotations

from pathlib import Path
from typing import Any


class Def:
    """The fundamental node type in MimIR. Everything in MimIR is a Def."""
    def type(self) -> Def:
        """Returns the type of this Def."""
        ...
    def var(self) -> Def:
        """Returns the variable (binder) associated with this Def if it's a lambda or continuation."""
        ...
    def proj(self, *args: int) -> Def:
        """Extracts the i-th projection from a tuple or extracts multiple projections."""
        ...
    def externalize(self) -> None:
        """Marks this Def as externally visible (e.g., exporting a function for C/LLVM)."""
        ...
    def dump(self) -> None:
        """Dumps the internal string representation of the Def to standard output."""
        ...
    def to_string(self) -> str:
        """Returns the string representation of the Def."""
        ...
    def value(self) -> int:
        """Returns the integer value of this Def if it is a literal."""
        ...
    def set(self, arg0: str) -> Def:
        """Sets the debug name of this Def."""
        ...
    def num_projs(self) -> int:
        """Returns the number of projections (elements) if this Def is a tuple."""
        ...
    def projs(self, arg0: int) -> list[Def]:
        """Returns a list of projections of a specified size."""
        ...
    def __iter__(self) -> Any:
        """Returns an iterator over the projections of this tuple/variable, allowing Python unpacking (e.g. )."""
        ...


class Lit(Def):
    """Represents a literal value in MimIR."""
    def value(self) -> int:
        """Returns the integer value of this literal."""
        ...


class Pi(Def):
    """Represents a function type in MimIR."""
    ...


class Lam(Def):
    """Represents a lambda (function or continuation) in MimIR."""
    def var(self) -> Def:
        """Returns the variable (arguments tuple) bound by this lambda."""
        ...
    def app(self, arg0: bool, arg1: Def, arg2: list[Def]) -> Lam:
        """
        Applies a function or continuation to a set of arguments.
        
        Args:
            arg0: Whether this is an implicit application.
            arg1: The callee (function or continuation) to apply.
            arg2: The arguments to pass to the callee.
        """
        ...
    def set(self, arg0: str) -> Lam:
        """Sets the debug name of this lambda."""
        ...

class Level:
    """Logging verbosity levels."""
    Error: Level
    Warn: Level
    Info: Level
    Verbose: Level
    Debug: Level


class Log:
    """Logging configuration for the MimIR driver."""
    def set_stdout(self) -> Log:
        """Enables logging to standard output."""
        ...
    def set(self, arg0: Level) -> Log:
        """Sets the logging verbosity level."""
        ...


class AST:
    """Abstract Syntax Tree for the Mim frontend language."""
    def __init__(self, arg0: World) -> None: ...


class Parser:
    """Parser for the Mim frontend language."""
    def __init__(self, arg0: AST) -> None: ...
    def plugin(self, arg0: str) -> None:
        """Loads and parses a plugin definition file."""
        ...


class PyParser:
    """Python interface to the Mim Parser."""
    def __init__(self, arg0: Parser) -> None: ...
    def plugin(self, arg0: str) -> None:
        """Loads and parses a plugin definition file."""
        ...


class World:
    """The central factory and arena for all MimIR nodes (Defs)."""
    def write(self) -> None: ...
    def annex(self, arg0: Any) -> Def: ...
    def annex_name(self, name: str) -> Def: ...
    def top_nat(self) -> Def:
        """Returns the Top (⊤) element of the Nat lattice, representing an unknown or unbounded size."""
        ...
    def type_bool(self) -> Def:
        """Returns the Bool type."""
        ...
    def type_i8(self) -> Def:
        """Returns the 8-bit integer type (I8), commonly used to represent characters."""
        ...
    def type_i32(self) -> Def:
        """Returns the 32-bit integer type (I32)."""
        ...
    def lit_nat_0(self) -> Lit:
        """Returns the natural number literal 0."""
        ...
    def lit_nat(self, arg0: int) -> Lit:
        """Creates a natural number literal with the given value."""
        ...
    def lit(self, arg0: Def, arg1: int) -> Lit:
        """Creates a literal of the specified type (arg0) with the given value (arg1)."""
        ...
    def lit_i8(self, arg0: int) -> Lit:
        """Creates an 8-bit integer literal with the given value."""
        ...
    def lit_i32(self, arg0: int) -> Lit:
        """Creates a 32-bit integer literal with the given value."""
        ...
    def type_idx(self, arg0: Def | int) -> Def:
        """Returns an index type bounded by the given size (e.g., Idx n)."""
        ...
    def lit_idx(self, arg0: int, arg1: int) -> Lit:
        """Creates an index literal with a specific bound (arg0) and value (arg1)."""
        ...
    def type_mem(self, addr_space: int = 0) -> Def:
        """Returns a Memory type `%mem.M` for tracking effects in the given address space."""
        ...
    def type_ptr(self, pointee: Def, ptr_sym: str = "%mem.Ptr0") -> Def:
        """Returns a pointer type `%mem.Ptr` or `%mem.Ptr0` pointing to the specified type in the given address space."""
        ...
    def cn(self, arg0: list[Def]) -> Def:
        """Creates a continuation (Cn) type taking the specified argument types."""
        ...
    def implicit_app(self, arg0: Def, arg1: list[Def]) -> Def:
        """Applies a function implicitly."""
        ...
    def app(self, arg0: Def, arg1: list[Def]) -> Def:
        """Applies a function explicitly to arguments."""
        ...
    def call(self, callee: str | Def, *stages: object, implicit: bool = False) -> Def:
        """
        Creates a call to an axiom or definition.
        
        Args:
            callee: The name of the axiom (e.g., '%math.arith.add') or the Def itself.
            stages: Arguments or implicit type parameters to provide to the call.
            implicit: Whether the call should be evaluated implicitly.
        """
        ...
    def tuple(self, arg0: list[Def]) -> Def:
        """Constructs a tuple from a list of Defs."""
        ...
    def bot(self, arg0: Def) -> Def:
        """Returns the Bottom (⊥) element for a given type."""
        ...
    def sym(self, arg0: str) -> Any: ...
    def mut_fun2(self, arg0: list[Def], arg1: list[Def]) -> Any:
        """Creates a mutable function with the given domain and codomain types."""
        ...
    def mut_fun(self, arg0: Def, arg1: list[Def]) -> Any: ...
    def arr(self, arg0: Def, arg1: Def) -> Def:
        """Creates an array type with a specific size and element type."""
        ...
    def optimize(self) -> None:
        """Runs the optimization pipeline on the world's graph."""
        ...
    def dot(self, arg0: str, arg1: bool, arg2: bool) -> None:
        """Exports the world graph in Graphviz DOT format."""
        ...
    def mut_con(self, arg0: list[Def]) -> Lam:
        """Creates a mutable continuation lambda with the specified argument types."""
        ...
    def annex_by_id(self, arg0: int) -> Def: ...


class Driver:
    """The main entry point for compiling and managing MimIR modules and plugins."""
    def __init__(self) -> None: ...
    def world(self) -> World:
        """Returns the World instance associated with this driver."""
        ...
    def add_import(self, path: str, name: str) -> Path: ...
    def add_search_path(self, path: Path) -> None:
        """Adds a path to the plugin and module search directories."""
        ...
    def log(self) -> Log:
        """Returns the logger for this driver."""
        ...
    def backend(self, backend: str, output_file_name: str, world: World) -> None:
        """
        Emits target code using the specified backend.
        
        Args:
            backend: The target backend name (e.g., "ll" for LLVM IR).
            output_file_name: The destination file path.
            world: The world to emit.
        """
        ...
    def load_plugins(self, plugins: list[str]) -> None:
        """Loads a list of C++ plugins."""
        ...
    def load_pluins(self, plugins: list[str]) -> None:
        """Typo fallback for load_plugins."""
        ...
