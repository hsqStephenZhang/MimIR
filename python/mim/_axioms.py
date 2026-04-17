from __future__ import annotations

from typing import Mapping, Sequence, TypeAlias

from ._axioms_generated import AXIOM_NAMESPACE_NAMES, AXIOM_TREES
from ._mim_core import Def, World

AxiomStage: TypeAlias = Def | Sequence[Def]


class AxiomNode:
    __slots__ = ("_symbol", "_children")

    def __init__(self, symbol: str | None, children: Mapping[str, "AxiomNode"]) -> None:
        self._symbol = symbol
        self._children = dict(children)

    def __call__(self, world: World, *stages: AxiomStage, implicit: bool = False) -> Def:
        if self._symbol is None:
            raise TypeError("namespace nodes are not directly callable")
        return world.call(self._symbol, *stages, implicit=implicit)

    def __getattr__(self, name: str) -> "AxiomNode":
        try:
            return self._children[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self) -> list[str]:
        return sorted(self._children)

    @property
    def symbol(self) -> str | None:
        return self._symbol


def _instantiate(tree: Mapping[str, object]) -> AxiomNode:
    children: dict[str, AxiomNode] = {}
    for name, val in tree.items():
        if isinstance(val, str):
            children[name] = AxiomNode(val, {})
        elif isinstance(val, dict):
            children[name] = _instantiate(val)
    return AxiomNode(None, children)


def _flatten_symbols(tree: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    for val in tree.values():
        if isinstance(val, str):
            result.append(val)
        elif isinstance(val, dict):
            result.extend(_flatten_symbols(val))
    return result


def list_axioms() -> dict[str, tuple[str, ...]]:
    return {plugin: tuple(sorted(_flatten_symbols(tree))) for plugin, tree in AXIOM_TREES.items()}


for _plugin, _tree in AXIOM_TREES.items():
    globals()["".join(part.capitalize() for part in _plugin.split("_"))] = _instantiate(_tree)


__all__ = ["AxiomNode", "AxiomStage", "AXIOM_NAMESPACE_NAMES", "list_axioms", *AXIOM_NAMESPACE_NAMES]
