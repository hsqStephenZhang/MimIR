from __future__ import annotations

from typing import Sequence, TypeAlias

from ._mim_core import Def, World

AxiomStage: TypeAlias = Def | Sequence[Def]


class AxiomNode:
    @property
    def symbol(self) -> str | None: ...
    def __call__(self, world: World, *stages: AxiomStage, implicit: bool = False) -> Def: ...
    def __getattr__(self, name: str) -> AxiomNode: ...


AXIOM_NAMESPACE_NAMES: tuple[str, ...]


def list_axioms() -> dict[str, tuple[str, ...]]: ...


Affine: AxiomNode
Autodiff: AxiomNode
Clos: AxiomNode
Compile: AxiomNode
Core: AxiomNode
Demo: AxiomNode
Direct: AxiomNode
Gpu: AxiomNode
Math: AxiomNode
Matrix: AxiomNode
Mem: AxiomNode
Opt: AxiomNode
Ord: AxiomNode
Refly: AxiomNode
Regex: AxiomNode
Tensor: AxiomNode
Tuple: AxiomNode
Vec: AxiomNode
