"""Likelihood objectives for continuous-time absorbing expression diffusion.

The primary objective in this module is a single, token-level hurdle negative
log likelihood (NLL).  A token is either exactly zero or positive.  The zero
event is modeled by a Bernoulli gate and positive values by a zero-truncated
Normal distribution.  Only diffusion-masked tokens contribute, and the
continuous-time absorbing schedule ``alpha(t) = 1 - t`` contributes its exact
``1 / t`` weight.

All likelihood arithmetic and reductions are deliberately performed in FP32,
even when model outputs are produced under mixed precision.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.config import NUM_GENES, LossConfig
from src.models.types import (
    HurdleDistributionParameters,
    TimeWeightedHurdleNLLOutput,
)
from src.utils.tensor_validation import (
    validate_diffusion_mask,
    validate_diffusion_time,
    validate_expression_tensor,
)


_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_PI_OVER_TWO = 0.5 * math.log(math.pi / 2.0)
_SQRT_TWO = math.sqrt(2.0)


def _validate_distribution_tensor(
    tensor: Tensor,
    *,
    name: str,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> None:
    """Validate one public hurdle-parameter tensor without changing it."""

    if not isinstance(tensor, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}."
        )
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}; "
            f"got {tuple(tensor.shape)}."
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype; got {tensor.dtype}.")
    if tensor.device != expected_device:
        raise ValueError(
            f"{name} and target must be on the same device; got "
            f"{tensor.device} and {expected_device}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def _zero_truncated_normal_nll(
    target: Tensor,
    location: Tensor,
    scale: Tensor,
) -> Tensor:
    r"""Return a stable FP32 ``-log Normal(x | mu,sigma,X>0)``.

    Directly evaluating

    .. math::

       \tfrac12(y-z)^2 + \log\sigma + \tfrac12\log(2\pi)
       + \log\Phi(z),\quad y=x/\sigma,\ z=\mu/\sigma,

    loses precision when ``z`` is very negative: its positive and negative
    ``z**2 / 2`` terms are individually huge.  For ``z < 0`` this function uses
    the exact identity

    .. math::

       \Phi(z)=\tfrac12\exp(-z^2/2)
       \operatorname{erfcx}(-z/\sqrt2)

    and analytically cancels those terms before floating-point evaluation:

    .. math::

       \tfrac12y^2-yz+\log\sigma+\tfrac12\log(\pi/2)
       +\log\operatorname{erfcx}(-z/\sqrt2).

    The ordinary ``log_ndtr`` expression remains well-conditioned for
    ``z >= 0``.  Safe placeholder values keep both vectorized branches finite
    before ``where`` selects the mathematically applicable result.
    """

    if target.dtype != torch.float32:
        raise TypeError("target must be FP32 inside truncated-Normal NLL.")
    if location.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError("location and scale must be FP32 inside the NLL.")

    standardized_target = target / scale
    standardized_location = location / scale
    negative_tail = standardized_location < 0.0

    tail_location = torch.where(
        negative_tail,
        standardized_location,
        torch.zeros_like(standardized_location),
    )
    tail_nll = (
        0.5 * standardized_target.square()
        - standardized_target * tail_location
        + torch.log(scale)
        + _HALF_LOG_PI_OVER_TWO
        + torch.log(torch.special.erfcx(-tail_location / _SQRT_TWO))
    )

    body_location = torch.where(
        negative_tail,
        torch.zeros_like(standardized_location),
        standardized_location,
    )
    body_nll = (
        0.5 * (standardized_target - body_location).square()
        + torch.log(scale)
        + _HALF_LOG_TWO_PI
        + torch.special.log_ndtr(body_location)
    )
    return torch.where(negative_tail, tail_nll, body_nll)


class TimeWeightedHurdleNLLLoss(nn.Module):
    r"""Compute the unified, inverse-time-weighted hurdle NLL.

    For target expression ``x >= 0`` and decoder outputs ``(a, mu, sigma)``,
    ``sigmoid(a)`` is the probability that the expression is positive.  The
    per-token NLL is

    .. math::

       1[x=0] softplus(a) + 1[x>0]\left(softplus(-a)
       - \log f_{TN+}(x; \mu, \sigma)\right),

    where ``f_TN+`` is a Normal density conditioned on being strictly positive.
    With a boolean diffusion mask ``M`` and per-cell time ``t``, the returned
    training scalar is

    .. math::

       L = (B G)^{-1} \sum_{b,i} M_{bi} t_b^{-1} NLL_{bi}.

    The denominator is always the fixed number of cell-gene positions, never
    the random masked-token count.  A row at ``t=0`` is valid only when it has
    no masked positions and contributes a differentiable zero.

    The returned sums are local sufficient statistics.  Under DDP, trainers
    must account for gradient averaging while using the global fixed
    normalizer; independently averaging rank-local losses is only equivalent
    when every rank has the same number of cells.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        expected = {
            "kind": "time_weighted_hurdle_nll",
            "reduction": "cell_gene_mean",
            "time_weighting": "inverse_t",
        }
        for field_name, expected_value in expected.items():
            actual_value = getattr(config, field_name, None)
            if actual_value != expected_value:
                raise ValueError(
                    f"TimeWeightedHurdleNLLLoss requires {field_name}="
                    f"{expected_value!r}, got {actual_value!r}."
                )
        self.config = config

    def forward(
        self,
        distribution_parameters: HurdleDistributionParameters,
        target: Tensor,
        diffusion_time: Tensor,
        diffusion_mask: Tensor,
    ) -> TimeWeightedHurdleNLLOutput:
        """Score hurdle parameters against clean ``target`` expression.

        Args:
            distribution_parameters: Strongly typed decoder tensors
                ``detection_logits``, ``positive_location`` and
                ``positive_scale``, each shaped ``[B, 19295, 1]``.  Positive
                scales must already include the decoder's minimum-scale floor.
            target: Finite, non-negative clean expression ``[B, 19295, 1]``.
            diffusion_time: FP32 continuous times ``[B]`` in ``[0, 1]``.
            diffusion_mask: Boolean ``[B, 19295]``; ``True`` means absorbing
                MASK and is the only kind of position scored by this loss.

        Returns:
            FP32 differentiable loss/sums plus integer sufficient statistics.
        """

        if not isinstance(distribution_parameters, HurdleDistributionParameters):
            raise TypeError(
                "distribution_parameters must be HurdleDistributionParameters, "
                f"got {type(distribution_parameters).__name__}."
            )

        validate_expression_tensor(
            target,
            num_genes=NUM_GENES,
            name="target",
            require_nonnegative=True,
        )
        batch_size = target.shape[0]
        if batch_size == 0:
            raise ValueError("target batch size must be positive.")
        validate_diffusion_time(
            diffusion_time,
            batch_size=batch_size,
            expected_device=target.device,
        )
        validate_diffusion_mask(
            diffusion_mask,
            batch_size=batch_size,
            num_genes=NUM_GENES,
            expected_device=target.device,
        )

        expected_shape = target.shape
        detection_logits = distribution_parameters.detection_logits
        positive_location = distribution_parameters.positive_location
        positive_scale = distribution_parameters.positive_scale
        _validate_distribution_tensor(
            detection_logits,
            name="detection_logits",
            expected_shape=expected_shape,
            expected_device=target.device,
        )
        _validate_distribution_tensor(
            positive_location,
            name="positive_location",
            expected_shape=expected_shape,
            expected_device=target.device,
        )
        _validate_distribution_tensor(
            positive_scale,
            name="positive_scale",
            expected_shape=expected_shape,
            expected_device=target.device,
        )
        if bool((positive_scale <= 0).any().item()):
            raise ValueError("positive_scale must be strictly positive.")

        zero_time = diffusion_time == 0.0
        masked_by_cell = diffusion_mask.any(dim=1)
        if bool((zero_time & masked_by_cell).any().item()):
            raise ValueError(
                "diffusion_time=0 is incompatible with masked positions in the "
                "same row."
            )

        # Cast before every nonlinear operation.  This avoids evaluating the
        # likelihood in FP16/BF16 under autocast and keeps all public sums FP32.
        target_fp32 = target.float()
        logits_fp32 = detection_logits.float()
        location_fp32 = positive_location.float()
        scale_fp32 = positive_scale.float()

        is_positive = target_fp32 > 0.0
        expanded_mask = diffusion_mask.unsqueeze(-1)
        masked_zero = expanded_mask & ~is_positive
        masked_positive = expanded_mask & is_positive

        # No epsilon or clipping is permitted in 1/t: doing so would change the
        # objective.  The safe branch solely defines valid, unmasked t=0 rows as
        # zero contribution without ever evaluating a reciprocal at zero.
        positive_time = diffusion_time > 0.0
        safe_time = torch.where(
            positive_time,
            diffusion_time,
            torch.ones_like(diffusion_time),
        )
        inverse_time = torch.where(
            positive_time,
            safe_time.reciprocal(),
            torch.zeros_like(safe_time),
        ).view(batch_size, 1, 1)

        # Evaluate each likelihood branch only on the tokens that use it.  In
        # addition to avoiding wasted work on a very long gene axis, this keeps
        # an irrelevant extreme positive-distribution parameter at a zero target
        # from producing ``0 * inf -> NaN`` in the zero branch.
        expanded_inverse_time = inverse_time.expand_as(target_fp32)
        zero_logits = logits_fp32.masked_select(masked_zero)
        zero_weights = expanded_inverse_time.masked_select(masked_zero)
        weighted_zero_nll_sum = (
            F.softplus(zero_logits) * zero_weights
        ).sum(dtype=torch.float32)

        positive_logits = logits_fp32.masked_select(masked_positive)
        positive_target = target_fp32.masked_select(masked_positive)
        positive_location_selected = location_fp32.masked_select(masked_positive)
        positive_scale_selected = scale_fp32.masked_select(masked_positive)
        positive_weights = expanded_inverse_time.masked_select(masked_positive)
        positive_value_nll = _zero_truncated_normal_nll(
            positive_target,
            positive_location_selected,
            positive_scale_selected,
        )
        positive_nll = F.softplus(-positive_logits) + positive_value_nll
        weighted_positive_nll_sum = (
            positive_nll * positive_weights
        ).sum(dtype=torch.float32)
        # Define the total from its two reported components so the accounting
        # identity is exact, including for empty selections.
        weighted_nll_sum = weighted_zero_nll_sum + weighted_positive_nll_sum

        cell_count = torch.tensor(
            batch_size,
            dtype=torch.int64,
            device=target.device,
        )
        normalizer = torch.tensor(
            batch_size * NUM_GENES,
            dtype=torch.int64,
            device=target.device,
        )
        masked_count = diffusion_mask.sum(dtype=torch.int64)
        masked_zero_count = masked_zero.sum(dtype=torch.int64)
        masked_positive_count = masked_positive.sum(dtype=torch.int64)

        loss = weighted_nll_sum / normalizer.to(dtype=torch.float32)

        return TimeWeightedHurdleNLLOutput(
            loss=loss,
            weighted_nll_sum=weighted_nll_sum,
            normalizer=normalizer,
            cell_count=cell_count,
            masked_count=masked_count,
            masked_zero_count=masked_zero_count,
            masked_positive_count=masked_positive_count,
            weighted_zero_nll_sum=weighted_zero_nll_sum,
            weighted_positive_nll_sum=weighted_positive_nll_sum,
        )
