from __future__ import annotations

from pathlib import Path
import subprocess

from ._mim_core import AST, Def, Driver, Lam, Level, Lit, Log, Parser, Pi, PyParser, World

__all__: list[str]


def plugin_search_paths() -> list[Path]: ...
def configure_driver(driver: Driver) -> Driver: ...
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
