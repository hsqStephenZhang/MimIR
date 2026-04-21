"""MimIR compiler backend for torch.compile.

Usage::

    import torch_backend  # registers "mimir" backend
    import torch

    @torch.compile(backend="mimir")
    def f(x, y):
        return x + y
"""

from ._backend import mimir_backend  # noqa: F401  (registers side-effectfully)

__all__ = ["mimir_backend"]
