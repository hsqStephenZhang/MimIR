"""Entry point: register the "mimir" torch.compile backend."""

import logging

import torch
import torch.fx as fx
from torch._dynamo.backends.registry import register_backend

from ._codegen import GraphCodegen, UnsupportedGraph

log = logging.getLogger(__name__)

# TODO: use ato_autograd to support backward and standard aten IR
@register_backend(name="mimir")
def mimir_backend(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> callable:
    """Compile an FX graph to native code via MimIR + LLVM."""
    try:
        return GraphCodegen(gm, example_inputs).compile()
    except UnsupportedGraph as e:
        log.info("MimIR backend: falling back to eager (%s)", e)
        return gm.forward
