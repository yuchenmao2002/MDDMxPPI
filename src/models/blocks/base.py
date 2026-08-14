"""Abstract block boundary shared by Performer and future hierarchical blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from src.models.types import BlockOutput, DenoiserContext


class DenoiserBlock(nn.Module, ABC):
    """One replaceable same-resolution block in the denoising backbone.

    Implementations may temporarily pool or construct internal hierarchies, but
    they must restore ``[B,G,d_model]`` before returning.  A block must not
    mutate its input/context, create a dense ``[B,G,G]`` tensor, or reinterpret
    the diffusion mask as padding.  Auxiliary losses are differentiable FP32
    scalars; diagnostics are detached scalars or small tensors only.
    """

    api_version = 1

    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,
        context: DenoiserContext,
        *,
        return_diagnostics: bool = False,
    ) -> BlockOutput:
        """Transform ``[B,G,d]`` hidden states without changing their shape."""

        raise NotImplementedError
