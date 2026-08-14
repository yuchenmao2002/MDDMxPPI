"""Reusable fail-fast validation for public model tensor contracts."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _require_tensor(value: Tensor, *, name: str) -> None:
    """Raise a useful error before accessing tensor-only attributes."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _require_device(
    tensor: Tensor,
    *,
    name: str,
    expected_device: Optional[torch.device],
) -> None:
    if expected_device is None:
        return
    try:
        normalized_device = torch.device(expected_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid expected_device {expected_device!r}.") from exc
    if tensor.device != normalized_device:
        raise ValueError(
            f"{name} must be on {normalized_device}, got device {tensor.device}."
        )


def _require_finite(tensor: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def validate_expression_tensor(
    expression_values: Tensor,
    *,
    num_genes: int,
    name: str = "expression_values",
    require_nonnegative: bool = True,
    expected_device: Optional[torch.device] = None,
) -> None:
    """Require finite floating ``[B,num_genes,1]`` expression values.

    Validation must not require nonnegative values for decoder predictions, but
    clean model inputs should be nonnegative in the configured processed space.
    Implementations may expose a separate flag when validating predictions.
    """

    _require_tensor(expression_values, name=name)
    _require_positive_int(num_genes, name="num_genes")

    expected_shape = f"[B,{num_genes},1]"
    if expression_values.ndim != 3:
        raise ValueError(
            f"{name} must have rank 3 and shape {expected_shape}; "
            f"got shape {tuple(expression_values.shape)}."
        )
    if expression_values.shape[1] != num_genes or expression_values.shape[2] != 1:
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"got shape {tuple(expression_values.shape)}."
        )
    if not expression_values.is_floating_point():
        raise TypeError(
            f"{name} must have a floating dtype; got {expression_values.dtype}."
        )
    _require_device(
        expression_values,
        name=name,
        expected_device=expected_device,
    )
    _require_finite(expression_values, name=name)
    if require_nonnegative and bool((expression_values < 0).any().item()):
        raise ValueError(
            f"{name} must be non-negative in the processed expression space."
        )


def validate_diffusion_time(
    diffusion_time: Tensor,
    *,
    batch_size: int,
    expected_device: Optional[torch.device] = None,
) -> None:
    """Require finite FP32 ``[B]`` values in the closed interval ``[0,1]``."""

    name = "diffusion_time"
    _require_tensor(diffusion_time, name=name)
    _require_nonnegative_int(batch_size, name="batch_size")
    if diffusion_time.ndim != 1 or diffusion_time.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have shape [{batch_size}]; "
            f"got shape {tuple(diffusion_time.shape)}."
        )
    if diffusion_time.dtype != torch.float32:
        raise TypeError(
            f"{name} must have dtype torch.float32; got {diffusion_time.dtype}."
        )
    _require_device(
        diffusion_time,
        name=name,
        expected_device=expected_device,
    )
    _require_finite(diffusion_time, name=name)
    if bool(((diffusion_time < 0.0) | (diffusion_time > 1.0)).any().item()):
        raise ValueError(f"{name} values must lie in the closed interval [0,1].")


def validate_diffusion_mask(
    diffusion_mask: Tensor,
    *,
    batch_size: int,
    num_genes: int,
    expected_device: Optional[torch.device] = None,
) -> None:
    """Require a boolean ``[B,num_genes]`` mask on the expected device."""

    name = "diffusion_mask"
    _require_tensor(diffusion_mask, name=name)
    _require_nonnegative_int(batch_size, name="batch_size")
    _require_positive_int(num_genes, name="num_genes")
    if diffusion_mask.ndim != 2 or tuple(diffusion_mask.shape) != (
        batch_size,
        num_genes,
    ):
        raise ValueError(
            f"{name} must have shape [{batch_size},{num_genes}]; "
            f"got shape {tuple(diffusion_mask.shape)}."
        )
    if diffusion_mask.dtype != torch.bool:
        raise TypeError(
            f"{name} must have dtype torch.bool; got {diffusion_mask.dtype}."
        )
    _require_device(
        diffusion_mask,
        name=name,
        expected_device=expected_device,
    )


def validate_hidden_states(
    hidden_states: Tensor,
    *,
    num_genes: int,
    d_model: int,
    name: str = "hidden_states",
    expected_device: Optional[torch.device] = None,
) -> None:
    """Require floating ``[B,num_genes,d_model]`` hidden states."""

    _require_tensor(hidden_states, name=name)
    _require_positive_int(num_genes, name="num_genes")
    _require_positive_int(d_model, name="d_model")
    expected_shape = f"[B,{num_genes},{d_model}]"
    if hidden_states.ndim != 3:
        raise ValueError(
            f"{name} must have rank 3 and shape {expected_shape}; "
            f"got shape {tuple(hidden_states.shape)}."
        )
    if hidden_states.shape[1] != num_genes or hidden_states.shape[2] != d_model:
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"got shape {tuple(hidden_states.shape)}."
        )
    if not hidden_states.is_floating_point():
        raise TypeError(
            f"{name} must have a floating dtype; got {hidden_states.dtype}."
        )
    _require_device(
        hidden_states,
        name=name,
        expected_device=expected_device,
    )
