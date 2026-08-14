"""Continuous-time absorbing-mask forward process and learnable MASK state.

The clean expression domain is continuous, while corruption introduces one
discrete absorbing symbol.  The state is represented by a numerical expression
tensor plus a separate boolean mask; no scalar sentinel is ever stored in the
expression tensor.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Generator, Tensor, nn

from src.models.config import NUM_GENES, ForwardProcessConfig
from src.models.types import ForwardProcessOutput
from src.utils.tensor_validation import (
    validate_diffusion_mask,
    validate_diffusion_time,
    validate_hidden_states,
)


class AbsorbingMaskForwardProcess:
    """Sample the configured marginal forward corruption.

    For each cell, sample ``t_b ~ Uniform[0,1)`` independently.  Conditional on
    that time, sample genes independently as ``M_bi ~ Bernoulli(t_b)``.  Do not
    force at least one mask.  Explicit caller-supplied times may include 1.0 so
    all-MASK validation and the future sampler can represent the endpoint.

    This object owns no learnable state and must not sample implicitly inside
    the core denoiser.
    """

    def __init__(self, config: ForwardProcessConfig) -> None:
        self.config = config

    def sample_times(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        """Return independent FP32 times with shape ``[B]`` in ``[0,1)``."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        return torch.rand(
            (batch_size,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

    def sample_mask(
        self,
        diffusion_time: Tensor,
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        """Return ``bool[B,19295]`` using per-gene independent Bernoulli draws."""

        if not isinstance(diffusion_time, Tensor):
            raise TypeError("diffusion_time must be a torch.Tensor.")
        if diffusion_time.ndim != 1:
            raise ValueError(
                "diffusion_time must have shape [B], got "
                f"{tuple(diffusion_time.shape)}."
            )
        validate_diffusion_time(
            diffusion_time,
            batch_size=diffusion_time.shape[0],
        )

        uniform_draws = torch.rand(
            (diffusion_time.shape[0], self.config.num_genes),
            device=diffusion_time.device,
            dtype=torch.float32,
            generator=generator,
        )
        diffusion_mask = uniform_draws < diffusion_time[:, None]
        validate_diffusion_mask(
            diffusion_mask,
            batch_size=diffusion_time.shape[0],
            num_genes=self.config.num_genes,
        )
        return diffusion_mask

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        diffusion_time: Optional[Tensor] = None,
        generator: Optional[Generator] = None,
    ) -> ForwardProcessOutput:
        """Return caller-supplied or sampled times together with a sampled mask."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        expected_device = torch.device(device)
        if diffusion_time is None:
            diffusion_time = self.sample_times(
                batch_size,
                device=expected_device,
                generator=generator,
            )
        else:
            validate_diffusion_time(diffusion_time, batch_size=batch_size)
            if diffusion_time.device != expected_device:
                raise ValueError(
                    "diffusion_time and requested sample device must match; got "
                    f"{diffusion_time.device} and {expected_device}."
                )

        diffusion_mask = self.sample_mask(diffusion_time, generator=generator)
        return ForwardProcessOutput(
            diffusion_time=diffusion_time,
            diffusion_mask=diffusion_mask,
        )


class AbsorbingStateEmbedding(nn.Module):
    """Replace encoded expression at MASK positions with one learned vector.

    The parameter has logical shape ``[512]`` and is shared by every gene and
    cell.  It is initialized from ``Normal(0, 0.02)`` using the caller's current
    PyTorch RNG state; construction does not alter or fix the global seed.
    Given ``E0:[B,G,d]`` and ``M:bool[B,G]``, return
    ``where(M[...,None], mask_embedding, E0)``.  Masked clean values must have no
    computational path to the result.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        self.d_model = d_model
        self.mask_embedding = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

    def forward(self, encoded_expression: Tensor, diffusion_mask: Tensor) -> Tensor:
        """Return masked expression embeddings with unchanged shape/dtype/device."""

        validate_hidden_states(
            encoded_expression,
            num_genes=NUM_GENES,
            d_model=self.d_model,
            name="encoded_expression",
        )
        validate_diffusion_mask(
            diffusion_mask,
            batch_size=encoded_expression.shape[0],
            num_genes=NUM_GENES,
        )
        if diffusion_mask.device != encoded_expression.device:
            raise ValueError(
                "diffusion_mask and encoded_expression must be on the same device; "
                f"got {diffusion_mask.device} and {encoded_expression.device}."
            )
        if self.mask_embedding.device != encoded_expression.device:
            raise ValueError(
                "mask_embedding and encoded_expression must be on the same device; "
                f"got {self.mask_embedding.device} and {encoded_expression.device}."
            )

        # Explicit dtype conversion preserves the activation dtype under mixed
        # precision while retaining the gradient path to the FP32 parameter.
        mask_embedding = self.mask_embedding.to(dtype=encoded_expression.dtype)
        return torch.where(
            diffusion_mask.unsqueeze(-1),
            mask_embedding.view(1, 1, self.d_model),
            encoded_expression,
        )
