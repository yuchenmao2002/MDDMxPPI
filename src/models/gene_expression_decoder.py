"""Shared hurdle truncated-Normal clean-expression readout head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.config import NUM_GENES, DecoderConfig
from src.models.types import DecoderOutput, HurdleDistributionParameters
from src.utils.tensor_validation import validate_hidden_states


_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_NEGATIVE_TAIL_CUTOFF = -10.0
_POSITIVE_TAIL_CUTOFF = 10.0


def _zero_truncated_normal_mean(location: Tensor, scale: Tensor) -> Tensor:
    """Return ``E[X | X>0]`` for ``X ~ Normal(location, scale)`` in FP32.

    The textbook expression ``location + scale * phi(z) / Phi(z)``, where
    ``z=location/scale``, catastrophically cancels for a far-negative location.
    Below ``z=-10`` we instead evaluate the inverse-Mills residual directly via
    its asymptotic series.  Above ``z=10`` the truncation correction is below
    FP32 relevance and the untruncated location is used.  The central branch
    uses ``torch.special.log_ndtr`` rather than taking a logarithm of a rounded
    CDF.  The final saturation is only relevant when the mathematical mean is
    outside the representable FP32 range.
    """

    if location.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError("Truncated-Normal mean inputs must be float32.")

    z = location / scale
    central_z = z.clamp(
        min=_NEGATIVE_TAIL_CUTOFF,
        max=_POSITIVE_TAIL_CUTOFF,
    )
    log_density = -0.5 * central_z.square() - _LOG_SQRT_TWO_PI
    inverse_mills = torch.exp(log_density - torch.special.log_ndtr(central_z))
    central_mean = scale * (central_z + inverse_mills)

    # phi(x) / Phi(-x) - x = 1/x - 2/x^3 + 10/x^5 - 74/x^7 + O(x^-9).
    # Evaluating it as an inverse-power polynomial avoids subtracting two
    # approximately equal values when location is far below zero.
    negative_x = (-z).clamp_min(1.0)
    inverse_x = negative_x.reciprocal()
    inverse_x_squared = inverse_x.square()
    negative_tail_residual = inverse_x * (
        1.0
        + inverse_x_squared
        * (
            -2.0
            + inverse_x_squared
            * (10.0 - 74.0 * inverse_x_squared)
        )
    )
    negative_tail_mean = scale * negative_tail_residual

    positive_mean = torch.where(
        z < _NEGATIVE_TAIL_CUTOFF,
        negative_tail_mean,
        central_mean,
    )
    positive_mean = torch.where(
        z > _POSITIVE_TAIL_CUTOFF,
        location,
        positive_mean,
    )

    float32_max = torch.finfo(torch.float32).max
    return torch.nan_to_num(
        positive_mean,
        nan=0.0,
        posinf=float32_max,
        neginf=0.0,
    ).clamp_(min=0.0, max=float32_max)


class GeneExpressionDecoder(nn.Module):
    """Parameterize one hurdle distribution independently at every gene token.

    The backbone already performs all cross-gene reasoning, so this head must
    not mix the gene axis or allocate gene-specific parameters.  One shared
    ``Linear(d_model,3)`` emits the positive/detection logit, underlying Normal
    location, and raw scale.  The projection and derived distribution parameters
    are computed in FP32 for likelihood and sampling stability.  The distribution
    mean is also FP32, but is evaluated only when the caller requests a point
    prediction.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(
            config.d_model,
            config.output_dim,
            bias=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        compute_point_prediction: bool = True,
    ) -> DecoderOutput:
        """Map ``[B,19295,d]`` to a typed hurdle distribution.

        Direct decoder and denoiser calls compute the hurdle mean by default.
        Likelihood-only training passes ``compute_point_prediction=False`` to
        avoid evaluating ``log_ndtr`` and the remaining mean arithmetic over
        every gene when the NLL consumes only distribution parameters.
        """

        if not isinstance(compute_point_prediction, bool):
            raise TypeError("compute_point_prediction must be a boolean.")

        validate_hidden_states(
            hidden_states,
            num_genes=NUM_GENES,
            d_model=self.config.d_model,
            expected_device=self.projection.weight.device,
        )
        # Casting only *after* ``nn.Linear`` would still let an enclosing AMP
        # context evaluate the projection in FP16/BF16.  Disable autocast for
        # the probability head itself so overflow or quantization in its three
        # raw parameters cannot be hidden by a later FP32 cast.  ``Tensor.float``
        # preserves autograd links to lower-precision parameters if a caller has
        # explicitly converted the module rather than using ordinary AMP.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            raw_parameters = F.linear(
                hidden_states.float(),
                self.projection.weight.float(),
                (
                    self.projection.bias.float()
                    if self.projection.bias is not None
                    else None
                ),
            )
        detection_logits, positive_location, raw_scale = raw_parameters.chunk(
            3,
            dim=-1,
        )
        positive_scale = self.config.min_scale + F.softplus(raw_scale)
        point_prediction = (
            torch.sigmoid(detection_logits)
            * _zero_truncated_normal_mean(
                positive_location,
                positive_scale,
            )
            if compute_point_prediction
            else None
        )

        parameters = HurdleDistributionParameters(
            detection_logits=detection_logits,
            positive_location=positive_location,
            positive_scale=positive_scale,
        )
        return DecoderOutput(
            point_prediction=point_prediction,
            distribution_parameters=parameters,
        )
