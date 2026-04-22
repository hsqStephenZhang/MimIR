from __future__ import annotations

from enum import Enum
from pathlib import Path
import subprocess

from ._axioms import (
    AXIOM_NAMESPACE_NAMES,
    AffineNode,
    AutodiffNode,
    AxiomNode,
    AxiomStage,
    BoundAxiomNode,
    BoundNamespaces,
    ClosNode,
    CompileNode,
    CoreNode,
    DemoNode,
    DirectNode,
    GpuNode,
    MathNode,
    MatrixNode,
    MemNode,
    OrdNode,
    ReflyNode,
    RegexNode,
    TensorNode,
    TupleNode,
    VecNode,
)
from ._mim_core import AST, Def, Driver, Lam, Level, Lit, Log, Parser, Pi, PyParser, World

__all__: list[str]

Affine: AffineNode
Autodiff: AutodiffNode
Clos: ClosNode
Compile: CompileNode
Core: CoreNode
Demo: DemoNode
Direct: DirectNode
Gpu: GpuNode
Math: MathNode
Matrix: MatrixNode
Mem: MemNode
Opt: AxiomNode
Ord: OrdNode
Refly: ReflyNode
Regex: RegexNode
Tensor: TensorNode
Tuple: TupleNode
Vec: VecNode


class Plugin(str, Enum):
    AFFINE: Plugin
    AUTODIFF: Plugin
    CLOS: Plugin
    COMPILE: Plugin
    CORE: Plugin
    DEMO: Plugin
    DIRECT: Plugin
    GPU: Plugin
    MATH: Plugin
    MATRIX: Plugin
    MEM: Plugin
    OPT: Plugin
    ORD: Plugin
    REFLY: Plugin
    REGEX: Plugin
    TENSOR: Plugin
    TUPLE: Plugin
    VEC: Plugin


PluginLike = str | Plugin


class DriverBuilder:
    def __init__(self) -> None: ...
    def plugin(self, plugin: PluginLike) -> DriverBuilder: ...
    def plugins(self, *plugins: PluginLike) -> DriverBuilder: ...
    def log_level(self, level: Level | None) -> DriverBuilder: ...
    def set_stdout(self, enabled: bool = True) -> DriverBuilder: ...
    def build(self) -> Driver: ...


def plugin_search_paths() -> list[Path]: ...
def configure_driver(driver: Driver) -> Driver: ...
def make_driver(*plugins: PluginLike, log_level: Level | None = None, set_stdout: bool = False) -> Driver: ...
def bind(world: World) -> BoundNamespaces: ...
def list_axioms() -> dict[str, tuple[str, ...]]: ...
def build_native_main_i32(world: World, result: Def, name: str = "main") -> Lam: ...
def emit_llvm(driver: Driver, world: World, output_path: str | Path) -> Path: ...
def clang_compile(
    llvm_ir: str | Path,
    output_path: str | Path,
    *,
    clang: str = "clang",
    extra_args: list[str] = ...,
) -> Path: ...
def build_native_executable(
    driver: Driver,
    world: World,
    stem: str | Path,
    *,
    clang: str = "clang",
    extra_args: list[str] = ...,
) -> Path: ...
def run_native_executable(
    executable: str | Path,
    args: list[str] = ...,
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]: ...
def matrix_i32(world: World, rows: list[list[int]]) -> Def: ...
def matrix_to_list(matrix: Def) -> list[list[int]]: ...
def transpose_2d(world: World, matrix: Def) -> Def: ...
def matmul_i32(world: World, lhs: Def, rhs: Def) -> Def: ...
