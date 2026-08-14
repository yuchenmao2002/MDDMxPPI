"""Continuous-expression absorbing diffusion model package.

The public API is intentionally assembled from small components.  The core
denoiser is deterministic and never samples diffusion time or masks inside its
``forward`` method.  Training-time corruption and loss computation are exposed
through a separate wrapper so that validation and the reverse sampler can
replay an exact state.  The decoder predicts a zero-versus-positive hurdle
distribution; training uses its inverse-time-weighted likelihood without
computing a point prediction.  Direct denoising can optionally report the
distribution expectation for metrics or deterministic imputation.
"""

from src.models.masked_diffusion_model import MaskedDiscreteDiffusionModel
from src.models.masked_expression_denoiser import MaskedExpressionDenoiser
from src.models.reverse_sampler import ReverseSampler, SamplingConfig

__all__ = [
    "MaskedDiscreteDiffusionModel",
    "MaskedExpressionDenoiser",
    "ReverseSampler",
    "SamplingConfig",
]
