"""Structured tensor contracts shared across model components.

All public model tensors are batch-first.  The gene axis is always present and
has length 19,295; no API silently squeezes a batch axis or the decoder's final
singleton channel.  ``diffusion_mask=True`` means that an expression value is
in the absorbing MASK state.  It never means padding and must not be inferred
from a numerical expression value of zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from torch import Tensor


@dataclass(frozen=True)
class DenoiserContext:
    """Context available to every interchangeable denoising block.

    The Performer deliberately ignores every field here: it does not consume
    ``diffusion_time`` and does not use ``diffusion_mask`` as an attention mask.
    The PPIL blocks do consume the derived mask-rate fields below, which is why
    they exist in the stable API.

    ``mask_rate`` and ``mask_rate_features`` are derived from ``diffusion_mask``
    once per forward pass by the backbone, because they are identical for every
    layer.  ``mask_rate`` is the realized fraction of genes still in the
    absorbing state; ``mask_rate_features`` is its Fourier representation
    ``[sin(2^0*pi*p), cos(2^0*pi*p), ..., sin(2^{h-1}*pi*p), cos(2^{h-1}*pi*p)]``
    with one band per attention head.  Both are ``None`` for backbones whose
    blocks do not read them, so those paths stay byte-for-byte unchanged.  They
    carry no gradient: they are a deterministic function of the boolean mask.
    """

    diffusion_time: Tensor  # float32 [B], values in [0,1]
    diffusion_mask: Tensor  # bool [B,G], True means absorbing MASK
    mask_rate: Optional[Tensor] = None  # float32 [B], realized masked fraction
    mask_rate_features: Optional[Tensor] = None  # float32 [B,2h], Fourier of mask_rate


@dataclass
class ForwardProcessOutput:
    """Sampled forward-process state; contains no corrupted numeric sentinel."""

    diffusion_time: Tensor  # float32 [B]
    diffusion_mask: Tensor  # bool [B,G]


@dataclass
class BlockOutput:
    """Output contract for one same-resolution denoising block."""

    hidden_states: Tensor  # [B,G,d_model], same external dtype/device as input
    aux_losses: Dict[str, Tensor] = field(default_factory=dict)
    diagnostics: Optional[Dict[str, Tensor]] = None


@dataclass
class BackboneOutput:
    """Output of an ordered stack of interchangeable denoising blocks."""

    last_hidden_state: Tensor  # [B,G,d_model]
    aux_losses: Dict[str, Tensor] = field(default_factory=dict)
    diagnostics: Optional[Dict[str, Tensor]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None


@dataclass
class HurdleDistributionParameters:
    """Parameters of the per-gene hurdle truncated-Normal distribution.

    Every tensor is FP32 with shape ``[B,G,1]`` and lives on the decoder input
    device.  ``detection_logits`` parameterizes the probability that expression
    is strictly positive.  ``positive_location`` is the location of the
    underlying Normal, not its truncated mean.  ``positive_scale`` has already
    undergone ``min_scale + softplus(raw_scale)`` and is strictly positive.
    """

    detection_logits: Tensor
    positive_location: Tensor
    positive_scale: Tensor


@dataclass
class DecoderOutput:
    """Probabilistic clean-expression decoder result.

    ``point_prediction`` is either the FP32 nonnegative finite
    hurdle-distribution mean with shape ``[B,G,1]`` or ``None`` when a caller
    needs only likelihood parameters.  Distribution parameters are always
    mandatory: likelihood training and reverse-process sampling consume the
    same probabilistic head rather than reconstructing parameters from the
    optional point estimate.
    """

    point_prediction: Optional[Tensor]  # FP32 [B,G,1], or None when not requested
    distribution_parameters: HurdleDistributionParameters


@dataclass
class ModelOutput:
    """Deterministic denoiser output for an explicitly supplied state.

    ``prediction`` mirrors ``decoder_output.point_prediction``.  Direct
    denoising computes it by default, while likelihood-only callers may
    explicitly omit it without omitting the distribution parameters.
    """

    prediction: Optional[Tensor]  # FP32 [B,G,1], or None when not requested
    decoder_output: DecoderOutput
    last_hidden_state: Optional[Tensor] = None
    aux_losses: Dict[str, Tensor] = field(default_factory=dict)
    diagnostics: Optional[Dict[str, Tensor]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None


@dataclass
class TimeWeightedHurdleNLLOutput:
    """Local sufficient statistics for inverse-time-weighted hurdle NLL.

    ``loss`` and all weighted sums are FP32 scalar tensors; the weighted sums
    remain differentiable.  ``normalizer`` is an int64 scalar containing the
    fixed local cell-gene count ``B*G`` rather than a random masked-token count.
    All other count fields are int64 scalar tensors.  Distributed training must
    globally reduce detached sufficient statistics and scale each rank's
    differentiable numerator against the global normalizer.
    """

    loss: Tensor
    weighted_nll_sum: Tensor
    normalizer: Tensor
    cell_count: Tensor
    masked_count: Tensor
    masked_zero_count: Tensor
    masked_positive_count: Tensor
    weighted_zero_nll_sum: Tensor
    weighted_positive_nll_sum: Tensor


@dataclass
class TrainingOutput:
    """Complete output of training-time corruption, denoising and scoring.

    ``loss`` and ``reconstruction_loss`` are the local FP32 hurdle NLL normalized
    by the fixed local cell-gene count.  The remaining scalar statistics follow
    :class:`TimeWeightedHurdleNLLOutput`: weighted sums are FP32 while the
    normalizer and counts are int64.  NLL training deliberately leaves
    ``prediction=None`` so it does not evaluate the comparatively expensive
    hurdle mean; diffusion time is FP32 ``[B]`` and the diffusion mask is bool
    ``[B,G]``.
    """

    loss: Tensor
    reconstruction_loss: Tensor
    weighted_nll_sum: Tensor
    normalizer: Tensor
    cell_count: Tensor
    masked_count: Tensor
    masked_zero_count: Tensor
    masked_positive_count: Tensor
    weighted_zero_nll_sum: Tensor
    weighted_positive_nll_sum: Tensor
    prediction: Optional[Tensor]
    diffusion_time: Tensor
    diffusion_mask: Tensor
    aux_losses: Dict[str, Tensor] = field(default_factory=dict)
    diagnostics: Optional[Dict[str, Tensor]] = None
