#!/usr/bin/env python3
"""Diagnose a bounded optimizer-step window of masked-diffusion training.

This is a deliberately separate entry point from ``train_masked_diffusion.py``.
It reuses that script's dataset, model, optimizer, scheduler, mixed-precision,
gradient-accumulation, and global-clipping contracts without changing the
production trainer. By default it records the first 500 *successful* optimizer
updates. ``--diagnostic-start-step N`` can instead execute N ordinary burn-in
updates and then record the following diagnostic window, without paying the
parameter-snapshot and detached-likelihood overhead during burn-in.

The diagnostic answers four concrete questions:

1. Which disjoint model module owns the pre-clipping gradient energy?
2. Does global clipping suppress unrelated modules because one shared module
   (especially the absorbing MASK embedding) has a large gradient?
3. Are large norms associated with the sampled continuous times or with the
   zero/positive Hurdle-NLL branches?
4. What parameter update did fused AdamW actually apply after clipping?

Inputs
------
The data/model/optimizer CLI is inherited from ``train_masked_diffusion.py``.
``--diagnostic-start-step`` is the number of successful optimizer updates to
complete without recording; ``--diagnostic-steps`` is the number to record.
Both count successful updates, not microbatches or optimizer attempts. The full
``--epochs`` value still defines the LR-scheduler horizon. Resume and
torch.compile are intentionally unsupported because this run is a from-scratch,
eager-mode instrumentation experiment.

Outputs
-------
``run_config.json``
    Auditable data/model/training and diagnostic configuration.
``gradient_steps.jsonl``
    Per-attempt records. Successful records contain exact loss sufficient
    statistics, time-bin statistics, pre-clipping module norms, clipping
    coefficient, and exact post-AdamW update-to-parameter ratios.
``summary.json``
    Quantiles, clipping rate, module gradient-energy shares, time correlations,
    aggregate time-bin losses, runtime, and peak allocated GPU memory.

No validation pass or model checkpoint is produced. Per-step synchronization,
extra detached likelihood accounting, and parameter snapshots make throughput
from this script unsuitable as a production-training benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Optional


# Match the production entry point: set this before importing anndata/h5py.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

import train_masked_diffusion as training
from src.models.config import NUM_GENES
from src.models.losses import _zero_truncated_normal_nll
from src.models.masked_diffusion_training import MaskedDiffusionTrainingModule
from src.models.types import HurdleDistributionParameters


SUFFICIENT_STATISTIC_FIELDS = (
    "weighted_nll_sum",
    "normalizer",
    "weighted_zero_nll_sum",
    "weighted_positive_nll_sum",
    "cell_count",
    "masked_count",
    "masked_zero_count",
    "masked_positive_count",
)

TIME_QUANTILE_FIELDS = (
    "min",
    "q01",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "q99",
    "max",
    "mean",
    "std",
    "max_inverse_time_with_mask",
    "inverse_weighted_mask_mass",
)

DECODER_CHANNEL_NAMES = (
    "detection",
    "positive_location",
    "positive_scale",
)


@dataclass(frozen=True)
class GradientGroup:
    """One disjoint trainable-parameter group used for module attribution."""

    name: str
    parameters: tuple[tuple[str, nn.Parameter], ...]

    @property
    def numel(self) -> int:
        return sum(parameter.numel() for _, parameter in self.parameters)


@dataclass(frozen=True)
class PreclipGroupTensors:
    """Device-resident reductions captured before global clipping."""

    gradient_squared_norm: Tensor
    parameter_squared_norm: Tensor
    max_abs_gradient: Tensor


class DiagnosticStop:
    """Signal state consumed only at complete optimizer-attempt boundaries."""

    requested = False
    signal_number: Optional[int] = None


DIAGNOSTIC_STOP = DiagnosticStop()


def request_stop(signum: int, _frame: object) -> None:
    DIAGNOSTIC_STOP.requested = True
    DIAGNOSTIC_STOP.signal_number = signum


def build_parser() -> argparse.ArgumentParser:
    """Extend the production parser with diagnostics-only controls."""

    parser = training.build_parser()
    parser.description = (
        "Run a from-scratch, single-GPU gradient diagnostic using the exact "
        "masked-diffusion training configuration."
    )
    # Validation/checkpointing are not part of this bounded diagnostic.
    parser.set_defaults(
        checkpoint_every=0,
        validate_every=0,
        early_stopping_patience=0,
    )
    diagnostic = parser.add_argument_group("gradient diagnostic")
    diagnostic.add_argument(
        "--diagnostic-start-step",
        type=int,
        default=0,
        help=(
            "Complete this many successful optimizer updates as an unrecorded "
            "burn-in, then start the diagnostic window. Zero preserves the "
            "historical behavior of recording from optimizer step 1."
        ),
    )
    diagnostic.add_argument(
        "--diagnostic-steps",
        type=int,
        default=500,
        help="Record this many successful optimizer updates after burn-in.",
    )
    diagnostic.add_argument(
        "--diagnostic-every",
        type=int,
        default=1,
        help=(
            "Write every Nth successful step to gradient_steps.jsonl. All "
            "successful steps still contribute to summary.json."
        ),
    )
    diagnostic.add_argument(
        "--time-bins",
        type=int,
        default=16,
        help="Number of equal-width [0,1] bins for detached loss accounting.",
    )
    return parser


def validate_diagnostic_args(args: argparse.Namespace) -> None:
    training.validate_args(args)
    if args.diagnostic_start_step < 0:
        raise ValueError("--diagnostic-start-step must be non-negative.")
    if args.diagnostic_steps <= 0:
        raise ValueError("--diagnostic-steps must be positive.")
    if args.diagnostic_every <= 0:
        raise ValueError("--diagnostic-every must be positive.")
    if not 2 <= args.time_bins <= 128:
        raise ValueError("--time-bins must lie in [2,128].")
    if args.resume is not None:
        raise ValueError(
            "Gradient diagnosis is intentionally from scratch; --resume is unsupported."
        )
    if args.torch_compile:
        raise ValueError(
            "Gradient diagnosis requires eager-mode parameter instrumentation; "
            "do not pass --torch-compile."
        )


def classify_parameter(name: str) -> str:
    """Map every trainable model parameter to exactly one primary module."""

    prefixes = (
        (
            "denoiser.absorbing_state_embedding.",
            "absorbing_mask_embedding",
        ),
        ("denoiser.decoder.", "decoder"),
        ("denoiser.gene_expression_encoder.", "expression_encoder"),
        ("denoiser.gene_identity_encoder.projection.", "identity_projection"),
        ("denoiser.backbone.", "performer_backbone"),
    )
    for prefix, group_name in prefixes:
        if name.startswith(prefix):
            return group_name
    raise ValueError(f"Unclassified trainable parameter: {name}")


def build_gradient_groups(model: nn.Module) -> tuple[GradientGroup, ...]:
    """Build disjoint groups and reject silent omissions or overlaps."""

    order = (
        "absorbing_mask_embedding",
        "decoder",
        "expression_encoder",
        "identity_projection",
        "performer_backbone",
    )
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        name: [] for name in order
    }
    trainable_names: list[str] = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_names.append(parameter_name)
        grouped[classify_parameter(parameter_name)].append(
            (parameter_name, parameter)
        )

    groups = tuple(
        GradientGroup(name=name, parameters=tuple(grouped[name])) for name in order
    )
    covered_names = [
        parameter_name
        for group in groups
        for parameter_name, _ in group.parameters
    ]
    if len(covered_names) != len(set(covered_names)):
        raise RuntimeError("A trainable parameter belongs to multiple groups.")
    if set(covered_names) != set(trainable_names):
        raise RuntimeError("Gradient groups do not exactly cover trainable parameters.")
    empty_groups = [group.name for group in groups if not group.parameters]
    if empty_groups:
        raise RuntimeError(f"Empty gradient groups: {', '.join(empty_groups)}")
    return groups


def pack_sufficient_statistics(output: Any) -> Tensor:
    """Pack one microbatch's exact scalar loss accounting on its device."""

    return torch.stack(
        [
            getattr(output, field).detach().to(dtype=torch.float64)
            for field in SUFFICIENT_STATISTIC_FIELDS
        ]
    )


@torch.no_grad()
def detached_time_bin_statistics(
    parameters: HurdleDistributionParameters,
    target: Tensor,
    diffusion_time: Tensor,
    diffusion_mask: Tensor,
    *,
    time_bins: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached Hurdle-NLL sufficient statistics by time bin.

    The differentiable production loss exposes only batch-level sufficient
    statistics.  This function reuses its stable positive-density primitive on
    detached decoder parameters, reduces first by cell, then assigns each cell
    to an equal-width time bin.  Counts are exact; NLL sums can differ from the
    production reduction by FP32 summation order. It never participates in
    backward, and the relative accounting difference is reported per step.

    Returns
    -------
    bin_statistics:
        Float64 tensor ``[8,time_bins]`` ordered like
        ``SUFFICIENT_STATISTIC_FIELDS``.
    masks_per_cell:
        Int64 ``[B]`` total masked genes.
    masked_positive_per_cell:
        Int64 ``[B]`` masked genes whose clean target is positive.
    """

    if target.shape[1:] != (NUM_GENES, 1):
        raise ValueError(
            f"Expected target [B,{NUM_GENES},1], got {tuple(target.shape)}."
        )

    clean = target.detach().squeeze(-1)
    mask = diffusion_mask.detach()
    times = diffusion_time.detach()
    logits = parameters.detection_logits.detach().squeeze(-1)
    locations = parameters.positive_location.detach().squeeze(-1)
    scales = parameters.positive_scale.detach().squeeze(-1)

    is_positive = clean > 0.0
    masked_positive = mask & is_positive
    masked_zero = mask & ~is_positive
    safe_time = torch.where(times > 0.0, times, torch.ones_like(times))
    inverse_time = torch.where(
        times > 0.0,
        safe_time.reciprocal(),
        torch.zeros_like(safe_time),
    )

    weighted_zero_per_cell = torch.zeros(
        target.shape[0],
        device=target.device,
        dtype=torch.float32,
    )
    zero_rows, zero_genes = masked_zero.nonzero(as_tuple=True)
    if zero_rows.numel() > 0:
        weighted_zero_per_cell.index_add_(
            0,
            zero_rows,
            F.softplus(logits[zero_rows, zero_genes])
            * inverse_time[zero_rows],
        )

    weighted_positive_per_cell = torch.zeros(
        target.shape[0],
        device=target.device,
        dtype=torch.float32,
    )
    positive_rows, positive_genes = masked_positive.nonzero(as_tuple=True)
    if positive_rows.numel() > 0:
        positive_nll = F.softplus(-logits[positive_rows, positive_genes])
        positive_nll = positive_nll + _zero_truncated_normal_nll(
            clean[positive_rows, positive_genes],
            locations[positive_rows, positive_genes],
            scales[positive_rows, positive_genes],
        )
        weighted_positive_per_cell.index_add_(
            0,
            positive_rows,
            positive_nll * inverse_time[positive_rows],
        )

    masks_per_cell = mask.sum(dim=1, dtype=torch.int64)
    masked_positive_per_cell = masked_positive.sum(dim=1, dtype=torch.int64)
    masked_zero_per_cell = masks_per_cell - masked_positive_per_cell
    bin_index = torch.clamp(
        torch.floor(times * time_bins).to(dtype=torch.int64),
        min=0,
        max=time_bins - 1,
    )

    def float_bin_sum(values: Tensor) -> Tensor:
        result = torch.zeros(
            time_bins,
            device=target.device,
            dtype=torch.float64,
        )
        result.index_add_(0, bin_index, values.to(dtype=torch.float64))
        return result

    def integer_bin_sum(values: Tensor) -> Tensor:
        result = torch.zeros(
            time_bins,
            device=target.device,
            dtype=torch.int64,
        )
        result.index_add_(0, bin_index, values)
        return result.to(dtype=torch.float64)

    cell_count = torch.bincount(bin_index, minlength=time_bins).to(
        dtype=torch.int64
    )
    weighted_zero = float_bin_sum(weighted_zero_per_cell)
    weighted_positive = float_bin_sum(weighted_positive_per_cell)
    bin_statistics = torch.stack(
        (
            weighted_zero + weighted_positive,
            (cell_count * NUM_GENES).to(dtype=torch.float64),
            weighted_zero,
            weighted_positive,
            cell_count.to(dtype=torch.float64),
            integer_bin_sum(masks_per_cell),
            integer_bin_sum(masked_zero_per_cell),
            integer_bin_sum(masked_positive_per_cell),
        )
    )
    return bin_statistics, masks_per_cell, masked_positive_per_cell


def collect_preclip_group_tensors(
    groups: tuple[GradientGroup, ...],
) -> dict[str, PreclipGroupTensors]:
    """Reduce parameter and unscaled-gradient statistics without host syncs."""

    result: dict[str, PreclipGroupTensors] = {}
    for group in groups:
        reference = group.parameters[0][1]
        gradient_squared = torch.zeros(
            (), device=reference.device, dtype=torch.float64
        )
        parameter_squared = torch.zeros_like(gradient_squared)
        max_abs_gradient = torch.zeros(
            (), device=reference.device, dtype=torch.float32
        )
        for _, parameter in group.parameters:
            parameter_value = parameter.detach().to(dtype=torch.float32)
            parameter_squared = parameter_squared + parameter_value.square().sum(
                dtype=torch.float64
            )
            gradient = parameter.grad
            if gradient is None:
                continue
            detached_gradient = gradient.detach().to(dtype=torch.float32)
            gradient_squared = gradient_squared + detached_gradient.square().sum(
                dtype=torch.float64
            )
            max_abs_gradient = torch.maximum(
                max_abs_gradient,
                detached_gradient.abs().max(),
            )
        result[group.name] = PreclipGroupTensors(
            gradient_squared_norm=gradient_squared,
            parameter_squared_norm=parameter_squared,
            max_abs_gradient=max_abs_gradient,
        )
    return result


def collect_decoder_channel_tensors(
    model: MaskedDiffusionTrainingModule,
) -> tuple[Tensor, Tensor]:
    """Return pre-clipping norm-squared/max-absolute values for three rows."""

    projection = model.denoiser.decoder.projection
    weight_gradient = projection.weight.grad
    bias_gradient = projection.bias.grad if projection.bias is not None else None
    if weight_gradient is None:
        raise RuntimeError("Decoder projection weight has no gradient.")
    gradient_squared = weight_gradient.detach().float().square().sum(
        dim=1, dtype=torch.float64
    )
    max_abs = weight_gradient.detach().float().abs().amax(dim=1)
    if bias_gradient is not None:
        bias = bias_gradient.detach().float()
        gradient_squared = gradient_squared + bias.square().to(dtype=torch.float64)
        max_abs = torch.maximum(max_abs, bias.abs())
    if gradient_squared.shape != (3,) or max_abs.shape != (3,):
        raise RuntimeError("Decoder projection no longer has exactly three channels.")
    return gradient_squared, max_abs


def snapshot_trainable_parameters(
    groups: tuple[GradientGroup, ...],
) -> dict[str, list[Tensor]]:
    """Clone parameters immediately before AdamW for exact update norms."""

    return {
        group.name: [
            parameter.detach().clone(memory_format=torch.preserve_format)
            for _, parameter in group.parameters
        ]
        for group in groups
    }


@torch.no_grad()
def collect_update_squared_norms(
    groups: tuple[GradientGroup, ...],
    snapshots: dict[str, list[Tensor]],
) -> dict[str, Tensor]:
    """Measure the actual fused-AdamW parameter delta for each module."""

    result: dict[str, Tensor] = {}
    for group in groups:
        before_values = snapshots[group.name]
        if len(before_values) != len(group.parameters):
            raise RuntimeError("Parameter snapshot no longer matches its group.")
        reference = group.parameters[0][1]
        squared_norm = torch.zeros(
            (), device=reference.device, dtype=torch.float64
        )
        for (_, parameter), before in zip(group.parameters, before_values):
            difference = parameter.detach().float() - before.float()
            squared_norm = squared_norm + difference.square().sum(
                dtype=torch.float64
            )
        result[group.name] = squared_norm
    return result


def time_feature_tensors(
    diffusion_times: Tensor,
    masks_per_cell: Tensor,
) -> Tensor:
    """Return a fixed-order device vector describing one effective batch."""

    probabilities = torch.tensor(
        (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0),
        device=diffusion_times.device,
        dtype=diffusion_times.dtype,
    )
    quantiles = torch.quantile(diffusion_times, probabilities)
    masked_rows = masks_per_cell > 0
    if bool(masked_rows.any().item()):
        max_inverse_time = diffusion_times[masked_rows].reciprocal().max()
    else:
        max_inverse_time = torch.zeros(
            (), device=diffusion_times.device, dtype=diffusion_times.dtype
        )
    safe_time = torch.where(
        diffusion_times > 0.0,
        diffusion_times,
        torch.ones_like(diffusion_times),
    )
    inverse_weighted_mask_mass = (
        masks_per_cell.float() / safe_time
    ).sum(dtype=torch.float64) / (diffusion_times.numel() * NUM_GENES)
    return torch.stack(
        (
            quantiles[0],
            quantiles[1],
            quantiles[2],
            quantiles[3],
            quantiles[4],
            quantiles[5],
            quantiles[6],
            quantiles[7],
            quantiles[8],
            diffusion_times.mean(),
            diffusion_times.std(unbiased=False),
            max_inverse_time,
            inverse_weighted_mask_mass.to(dtype=diffusion_times.dtype),
        )
    )


def sufficient_statistics_dict(values: list[float]) -> dict[str, Any]:
    if len(values) != len(SUFFICIENT_STATISTIC_FIELDS):
        raise ValueError("Unexpected sufficient-statistics vector length.")
    result = dict(zip(SUFFICIENT_STATISTIC_FIELDS, values))
    for field in (
        "normalizer",
        "cell_count",
        "masked_count",
        "masked_zero_count",
        "masked_positive_count",
    ):
        result[field] = int(round(result[field]))
    normalizer = result["normalizer"]
    result["loss"] = (
        result["weighted_nll_sum"] / normalizer if normalizer > 0 else None
    )
    result["zero_loss"] = (
        result["weighted_zero_nll_sum"] / normalizer
        if normalizer > 0
        else None
    )
    result["positive_loss"] = (
        result["weighted_positive_nll_sum"] / normalizer
        if normalizer > 0
        else None
    )
    return result


def time_bin_records(
    values: list[float],
    *,
    time_bins: int,
) -> list[dict[str, Any]]:
    expected = len(SUFFICIENT_STATISTIC_FIELDS) * time_bins
    if len(values) != expected:
        raise ValueError(f"Expected {expected} time-bin values, got {len(values)}.")
    matrix = np.asarray(values, dtype=np.float64).reshape(
        len(SUFFICIENT_STATISTIC_FIELDS), time_bins
    )
    records: list[dict[str, Any]] = []
    for bin_index in range(time_bins):
        statistics = sufficient_statistics_dict(matrix[:, bin_index].tolist())
        records.append(
            {
                "bin": bin_index,
                "left": bin_index / time_bins,
                "right": (bin_index + 1) / time_bins,
                **statistics,
            }
        )
    return records


def materialize_successful_step(
    *,
    groups: tuple[GradientGroup, ...],
    preclip: dict[str, PreclipGroupTensors],
    update_squared: dict[str, Tensor],
    decoder_channel_squared: Tensor,
    decoder_channel_max_abs: Tensor,
    returned_global_norm: Tensor,
    sufficient_statistics: Tensor,
    time_bin_statistics: Tensor,
    time_features: Tensor,
    time_bins: int,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Transfer all per-step GPU reductions in one packed synchronization."""

    group_rows = torch.stack(
        [
            torch.stack(
                (
                    preclip[group.name].gradient_squared_norm,
                    preclip[group.name].parameter_squared_norm,
                    preclip[group.name].max_abs_gradient.to(dtype=torch.float64),
                    update_squared[group.name],
                )
            )
            for group in groups
        ]
    )
    pieces = (
        group_rows.reshape(-1),
        decoder_channel_squared.to(dtype=torch.float64),
        decoder_channel_max_abs.to(dtype=torch.float64),
        returned_global_norm.detach().reshape(1).to(dtype=torch.float64),
        sufficient_statistics.detach().reshape(-1).to(dtype=torch.float64),
        time_bin_statistics.detach().reshape(-1).to(dtype=torch.float64),
        time_features.detach().reshape(-1).to(dtype=torch.float64),
    )
    packed = torch.cat(pieces).cpu().tolist()
    offset = 0

    module_metrics: dict[str, dict[str, Any]] = {}
    total_squared = 0.0
    raw_group_values: dict[str, tuple[float, float, float, float]] = {}
    for group in groups:
        row = tuple(float(value) for value in packed[offset : offset + 4])
        offset += 4
        raw_group_values[group.name] = row
        total_squared += row[0]

    for group in groups:
        gradient_squared, parameter_squared, max_abs, update_sq = raw_group_values[
            group.name
        ]
        parameter_norm = math.sqrt(max(0.0, parameter_squared))
        update_norm = math.sqrt(max(0.0, update_sq))
        module_metrics[group.name] = {
            "numel": group.numel,
            "parameter_norm": parameter_norm,
            "preclip_gradient_norm": math.sqrt(max(0.0, gradient_squared)),
            "preclip_gradient_rms": math.sqrt(
                max(0.0, gradient_squared) / group.numel
            ),
            "preclip_gradient_max_abs": max_abs,
            "gradient_energy_fraction": (
                gradient_squared / total_squared if total_squared > 0.0 else None
            ),
            "update_norm": update_norm,
            "update_to_parameter_ratio": (
                update_norm / parameter_norm if parameter_norm > 0.0 else None
            ),
        }

    decoder_squared = packed[offset : offset + 3]
    offset += 3
    decoder_max_abs = packed[offset : offset + 3]
    offset += 3
    decoder_channels = {
        channel_name: {
            "preclip_gradient_norm": math.sqrt(max(0.0, decoder_squared[index])),
            "preclip_gradient_max_abs": decoder_max_abs[index],
        }
        for index, channel_name in enumerate(DECODER_CHANNEL_NAMES)
    }

    clip_returned_norm = float(packed[offset])
    offset += 1
    statistic_count = len(SUFFICIENT_STATISTIC_FIELDS)
    loss_statistics = sufficient_statistics_dict(
        packed[offset : offset + statistic_count]
    )
    offset += statistic_count
    bin_value_count = statistic_count * time_bins
    bins = time_bin_records(
        packed[offset : offset + bin_value_count],
        time_bins=time_bins,
    )
    offset += bin_value_count
    time_values = packed[offset : offset + len(TIME_QUANTILE_FIELDS)]
    offset += len(TIME_QUANTILE_FIELDS)
    if offset != len(packed):
        raise RuntimeError("Packed diagnostic vector was not consumed exactly.")

    recomputed_global_norm = math.sqrt(max(0.0, total_squared))
    denominator = max(1.0, abs(clip_returned_norm))
    norm_reconstruction_error = (
        abs(recomputed_global_norm - clip_returned_norm) / denominator
    )
    # clip_grad_norm_ reduces per-parameter FP32 norms, whereas the diagnostic
    # directly accumulates FP64 squared elements by module. Small differences in
    # reduction order are expected for roughly 20M parameters.
    if norm_reconstruction_error > 1e-3:
        raise RuntimeError(
            "Module gradient norms do not reconstruct clip_grad_norm_'s global norm."
        )
    time_metrics = dict(zip(TIME_QUANTILE_FIELDS, time_values))
    bin_weighted_sum = sum(record["weighted_nll_sum"] for record in bins)
    consistency_error = abs(
        bin_weighted_sum - loss_statistics["weighted_nll_sum"]
    ) / max(1.0, abs(loss_statistics["weighted_nll_sum"]))

    return {
        "loss": loss_statistics,
        "time": time_metrics,
        "time_bins": bins,
        "time_bin_relative_nll_accounting_error": consistency_error,
        "preclip_global_grad_norm": clip_returned_norm,
        "reconstructed_preclip_global_grad_norm": recomputed_global_norm,
        "global_norm_relative_reconstruction_error": norm_reconstruction_error,
        "max_grad_norm": max_grad_norm,
        "clipped": clip_returned_norm > max_grad_norm,
        "clip_coefficient": min(
            1.0,
            max_grad_norm / (clip_returned_norm + 1e-6),
        ),
        "modules": module_metrics,
        "decoder_channels": decoder_channels,
    }


def describe(values: list[float]) -> dict[str, float]:
    """Return stable scalar distribution summaries for JSON output."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Diagnostic summary received empty or non-finite values.")
    quantiles = np.quantile(array, (0.50, 0.75, 0.90, 0.95, 0.99))
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(array.max()),
    }


def correlation(left: list[float], right: list[float]) -> Optional[float]:
    """Compute Pearson correlation, returning None for a constant vector."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.size != right_array.size or left_array.size < 2:
        return None
    if left_array.std() == 0.0 or right_array.std() == 0.0:
        return None
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    return value if math.isfinite(value) else None


def build_summary(
    *,
    records: list[dict[str, Any]],
    groups: tuple[GradientGroup, ...],
    aggregate_time_bins: np.ndarray,
    recorded_attempted_steps: int,
    recorded_skipped_steps: int,
    target_recorded_steps: int,
    diagnostic_start_step: int,
    total_successful_steps: int,
    total_attempted_steps: int,
    total_skipped_steps: int,
    burn_in_attempted_steps: int,
    burn_in_skipped_steps: int,
    recorded_elapsed_seconds: float,
    burn_in_elapsed_seconds: float,
    total_elapsed_seconds: float,
    recorded_peak_gpu_memory_bytes: int,
    burn_in_peak_gpu_memory_bytes: int,
    completion_reason: str,
) -> dict[str, Any]:
    """Aggregate per-step records without treating bins as equal-size means."""

    first_recorded_step = diagnostic_start_step + 1
    last_target_recorded_step = diagnostic_start_step + target_recorded_steps
    summary: dict[str, Any] = {
        "event": "gradient_diagnostic_summary",
        "completion_reason": completion_reason,
        # Historical fields continue to describe the recorded diagnostic window.
        "target_optimizer_steps": target_recorded_steps,
        "successful_optimizer_steps": len(records),
        "attempted_optimizer_steps": recorded_attempted_steps,
        "skipped_optimizer_steps": recorded_skipped_steps,
        "diagnostic_start_step": diagnostic_start_step,
        "recorded_window": {
            "first_target_global_optimizer_step": first_recorded_step,
            "last_target_global_optimizer_step": last_target_recorded_step,
            "first_recorded_global_optimizer_step": (
                records[0]["global_optimizer_step"] if records else None
            ),
            "last_recorded_global_optimizer_step": (
                records[-1]["global_optimizer_step"] if records else None
            ),
            "target_successful_optimizer_steps": target_recorded_steps,
            "successful_optimizer_steps": len(records),
            "attempted_optimizer_steps": recorded_attempted_steps,
            "skipped_optimizer_steps": recorded_skipped_steps,
            "elapsed_seconds": recorded_elapsed_seconds,
            "peak_gpu_memory_bytes": recorded_peak_gpu_memory_bytes,
        },
        "burn_in": {
            "target_successful_optimizer_steps": diagnostic_start_step,
            "successful_optimizer_steps": min(
                total_successful_steps, diagnostic_start_step
            ),
            "attempted_optimizer_steps": burn_in_attempted_steps,
            "skipped_optimizer_steps": burn_in_skipped_steps,
            "elapsed_seconds": burn_in_elapsed_seconds,
            "peak_gpu_memory_bytes": burn_in_peak_gpu_memory_bytes,
            "heavy_diagnostics_enabled": False,
        },
        "total_execution": {
            "target_successful_optimizer_steps": last_target_recorded_step,
            "successful_optimizer_steps": total_successful_steps,
            "attempted_optimizer_steps": total_attempted_steps,
            "skipped_optimizer_steps": total_skipped_steps,
            "elapsed_seconds": total_elapsed_seconds,
            "peak_gpu_memory_bytes": max(
                burn_in_peak_gpu_memory_bytes,
                recorded_peak_gpu_memory_bytes,
            ),
        },
        # Keep the legacy duration/memory keys scoped to the measured window.
        "elapsed_seconds": recorded_elapsed_seconds,
        "peak_gpu_memory_bytes": recorded_peak_gpu_memory_bytes,
        "throughput_warning": (
            "Recorded-window throughput includes per-step synchronization, "
            "detached time-bin likelihood accounting, and exact parameter "
            "snapshots, so it is unsuitable for production throughput comparison."
        ),
    }
    if not records:
        summary.update(
            {
                "clip_count": 0,
                "clip_rate": None,
                "preclip_global_grad_norm": None,
                "loss": None,
                "modules": {},
                "decoder_channels": {},
                "time_correlations": {
                    "grad_norm_vs_min_time": None,
                    "grad_norm_vs_max_inverse_time_with_mask": None,
                    "grad_norm_vs_inverse_weighted_mask_mass": None,
                    "grad_norm_vs_positive_loss": None,
                },
                "aggregate_time_bins": [],
                "diagnostic_cells_per_second": None,
            }
        )
        return summary

    global_norms = [record["preclip_global_grad_norm"] for record in records]
    module_summary: dict[str, Any] = {}
    global_energy = sum(value * value for value in global_norms)
    for group in groups:
        gradient_norms = [
            record["modules"][group.name]["preclip_gradient_norm"]
            for record in records
        ]
        update_ratios = [
            record["modules"][group.name]["update_to_parameter_ratio"]
            for record in records
        ]
        module_energy = sum(value * value for value in gradient_norms)
        module_summary[group.name] = {
            "numel": group.numel,
            "preclip_gradient_norm": describe(gradient_norms),
            "update_to_parameter_ratio": describe(update_ratios),
            "aggregate_gradient_energy_fraction": (
                module_energy / global_energy if global_energy > 0.0 else None
            ),
        }

    channel_summary = {
        channel_name: describe(
            [
                record["decoder_channels"][channel_name][
                    "preclip_gradient_norm"
                ]
                for record in records
            ]
        )
        for channel_name in DECODER_CHANNEL_NAMES
    }
    aggregate_bin_records = time_bin_records(
        aggregate_time_bins.reshape(-1).tolist(),
        time_bins=aggregate_time_bins.shape[1],
    )
    loss_values = [record["loss"]["loss"] for record in records]
    positive_loss_values = [
        record["loss"]["positive_loss"] for record in records
    ]
    minimum_times = [record["time"]["min"] for record in records]
    maximum_inverse_times = [
        record["time"]["max_inverse_time_with_mask"] for record in records
    ]
    inverse_mask_mass = [
        record["time"]["inverse_weighted_mask_mass"] for record in records
    ]
    summary.update(
        {
            "clip_count": sum(bool(record["clipped"]) for record in records),
            "clip_rate": sum(bool(record["clipped"]) for record in records)
            / len(records),
            "preclip_global_grad_norm": describe(global_norms),
            "loss": describe(loss_values),
            "modules": module_summary,
            "decoder_channels": channel_summary,
            "time_correlations": {
                "grad_norm_vs_min_time": correlation(global_norms, minimum_times),
                "grad_norm_vs_max_inverse_time_with_mask": correlation(
                    global_norms, maximum_inverse_times
                ),
                "grad_norm_vs_inverse_weighted_mask_mass": correlation(
                    global_norms, inverse_mask_mass
                ),
                "grad_norm_vs_positive_loss": correlation(
                    global_norms, positive_loss_values
                ),
            },
            "aggregate_time_bins": aggregate_bin_records,
            "diagnostic_cells_per_second": (
                sum(record["group_cells"] for record in records)
                / recorded_elapsed_seconds
                if recorded_elapsed_seconds > 0.0
                else None
            ),
        }
    )
    return summary


def resolve_paths(args: argparse.Namespace) -> None:
    args.data_path = training.resolve_input(args.data_path, "Data h5ad")
    args.gene_mapping_path = training.resolve_input(
        args.gene_mapping_path, "Gene mapping"
    )
    args.gene_weights_path = training.resolve_input(
        args.gene_weights_path, "Gene weights"
    )
    args.gene_manifest_path = training.resolve_input(
        args.gene_manifest_path, "Gene embedding manifest"
    )
    args.output_dir = args.output_dir.expanduser().resolve()


def main() -> int:
    args = build_parser().parse_args()
    validate_diagnostic_args(args)
    resolve_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "gradient_steps.jsonl"
    summary_path = args.output_dir / "summary.json"
    run_config_path = args.output_dir / "run_config.json"
    if any(path.exists() for path in (metrics_path, summary_path, run_config_path)):
        raise FileExistsError(
            "Output directory already contains a gradient diagnostic; choose a "
            "new --output-dir."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Gradient diagnosis requires one CUDA GPU.")
    if torch.cuda.device_count() != 1:
        print(
            f"WARNING: {torch.cuda.device_count()} GPUs are visible; only cuda:0 is used.",
            flush=True,
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    training.set_global_seed(args.seed)
    autocast_enabled, autocast_dtype = training.amp_settings(args.precision)

    expected_genes = training.read_expected_genes(
        args.gene_mapping_path,
        index_column=args.gene_index_column,
        name_column=args.gene_name_column,
    )
    metadata = ad.read_h5ad(args.data_path, backed="r")
    try:
        n_obs = int(metadata.n_obs)
    finally:
        training.close_backed(metadata)
    train_indices, val_indices = training.deterministic_split(
        n_obs,
        args.val_fraction,
        args.seed,
        args.max_train_cells,
        args.max_val_cells,
    )
    train_dataset = training.BackedH5adRowDataset(
        args.data_path,
        args.matrix_key,
        train_indices,
        expected_genes,
    )
    sampler_generator = torch.Generator(device="cpu")
    worker_generator = torch.Generator(device="cpu")
    worker_generator.manual_seed(args.seed + 200_000)
    train_loader = training.make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        sampler_generator=sampler_generator,
        worker_generator=worker_generator,
    )

    data_stat = args.data_path.stat()
    data_contract = {
        "path": str(args.data_path),
        "file_size": data_stat.st_size,
        "file_mtime_ns": data_stat.st_mtime_ns,
        "matrix_key": args.matrix_key,
        "matrix_dtype": train_dataset.matrix_dtype,
        "n_obs": n_obs,
        "n_vars": NUM_GENES,
        "train_cells": len(train_dataset),
        "validation_cells": len(val_indices),
        "gene_order_sha256": training.sha256_strings(expected_genes),
        "split_sha256": training.sha256_indices(train_indices, val_indices),
    }

    model_config = training.build_model_config(args)
    model = MaskedDiffusionTrainingModule.from_config(model_config).to(device)
    groups = build_gradient_groups(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(group.numel for group in groups)
    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    try:
        optimizer = AdamW(model.parameters(), fused=True, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    updates_per_epoch = math.ceil(
        len(train_loader) / args.grad_accumulation_steps
    )
    total_scheduler_steps = args.epochs * updates_per_epoch
    total_requested_steps = (
        args.diagnostic_start_step + args.diagnostic_steps
    )
    if total_requested_steps > total_scheduler_steps:
        raise ValueError(
            "--diagnostic-start-step + --diagnostic-steps exceeds the complete "
            "scheduler horizon defined by --epochs and the selected dataset "
            f"({total_requested_steps} requested, {total_scheduler_steps} available)."
        )
    scheduler = training.build_scheduler(
        optimizer,
        total_steps=total_scheduler_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    diffusion_generator = torch.Generator(device=device)
    diffusion_generator.manual_seed(args.seed + 1)

    run_config = {
        "event": "gradient_diagnostic_config",
        "architecture_version": model_config.architecture_version,
        "arguments": vars(args),
        "model_config": asdict(model_config),
        "data_contract": data_contract,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "gradient_groups": {
            group.name: {
                "numel": group.numel,
                "parameter_names": [name for name, _ in group.parameters],
            }
            for group in groups
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "updates_per_epoch": updates_per_epoch,
        "total_scheduler_steps": total_scheduler_steps,
        "diagnostic_contract": {
            "diagnostic_start_step_semantics": (
                "number of successful optimizer updates completed as unrecorded "
                "burn-in; the first recorded global optimizer step is start + 1"
            ),
            "burn_in_successful_optimizer_steps": args.diagnostic_start_step,
            "recorded_successful_optimizer_steps": args.diagnostic_steps,
            "first_recorded_global_optimizer_step": (
                args.diagnostic_start_step + 1
            ),
            "last_recorded_global_optimizer_step": total_requested_steps,
            "stop_after_total_successful_optimizer_steps": total_requested_steps,
            "burn_in_instrumentation": (
                "training forward/backward, global clipping, AdamW, and scheduler "
                "plus the production-style scalar finite-loss guard; no detached "
                "time-bin accounting, parameter snapshots, or per-step diagnostic "
                "materialization/synchronization; one CUDA synchronization "
                "separates burn-in and recorded timing"
            ),
            "burn_in_numerical_safety": (
                "no per-microbatch host loss synchronization; BF16/FP32 fail on "
                "a non-finite global gradient norm at every optimizer attempt, "
                "while FP16 GradScaler skips and counts overflowed attempts"
            ),
            "preclip_measurement": "after GradScaler.unscale_, before global clipping",
            "update_measurement": "exact parameter delta after AdamW.step",
            "time_bins": args.time_bins,
            "validation": "disabled",
            "checkpoint": "disabled",
            "throughput_is_representative": False,
        },
    }
    training.atomic_json_dump(run_config, run_config_path)
    print(json.dumps(training.jsonable(run_config), sort_keys=True), flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    successful_records: list[dict[str, Any]] = []
    total_successful_steps = 0
    total_attempted_steps = 0
    total_skipped_steps = 0
    burn_in_attempted_steps = 0
    burn_in_skipped_steps = 0
    recorded_attempted_steps = 0
    recorded_skipped_steps = 0
    aggregate_time_bins = np.zeros(
        (len(SUFFICIENT_STATISTIC_FIELDS), args.time_bins),
        dtype=np.float64,
    )
    start_time = time.monotonic()
    recorded_start_time: Optional[float] = (
        start_time if args.diagnostic_start_step == 0 else None
    )
    burn_in_elapsed_seconds: Optional[float] = (
        0.0 if args.diagnostic_start_step == 0 else None
    )
    burn_in_peak_gpu_memory_bytes = 0
    completion_reason = "scheduler_horizon_exhausted"

    group_sufficient: list[Tensor] = []
    group_time_bins: list[Tensor] = []
    group_times: list[Tensor] = []
    group_mask_counts: list[Tensor] = []
    group_microbatches = 0
    group_started_at = start_time
    group_is_recorded = args.diagnostic_start_step == 0

    for epoch in range(args.epochs):
        sampler_generator.manual_seed(args.seed + epoch)
        num_batches = len(train_loader)
        for batch_index, clean_expression in enumerate(train_loader):
            if group_microbatches == 0:
                group_started_at = time.monotonic()
                group_is_recorded = (
                    total_successful_steps >= args.diagnostic_start_step
                )
            microbatch_cells = int(clean_expression.shape[0])
            clean_expression = clean_expression.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            group_start = (
                batch_index // args.grad_accumulation_steps
            ) * args.grad_accumulation_steps
            group_stop = min(
                group_start + args.grad_accumulation_steps,
                num_batches,
            )
            nominal_batch_size = train_loader.batch_size
            if not isinstance(nominal_batch_size, int):
                raise RuntimeError("Training DataLoader must have a fixed batch size.")
            group_cells = nominal_batch_size * (group_stop - group_start)
            if group_stop == num_batches:
                final_batch_cells = len(train_loader.dataset) - nominal_batch_size * (
                    num_batches - 1
                )
                group_cells += final_batch_cells - nominal_batch_size
            if group_cells <= 0:
                raise RuntimeError("Gradient-accumulation group contains no cells.")

            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                forward_state = model.forward_process.sample(
                    microbatch_cells,
                    device=device,
                    generator=diffusion_generator,
                )
                denoiser_output = model.denoiser(
                    clean_expression,
                    forward_state.diffusion_time,
                    forward_state.diffusion_mask,
                    return_hidden_state=False,
                    output_hidden_states=False,
                    return_diagnostics=False,
                    compute_point_prediction=False,
                )
                distribution_parameters = (
                    denoiser_output.decoder_output.distribution_parameters
                )
                reconstruction = model.reconstruction_loss(
                    distribution_parameters,
                    clean_expression,
                    forward_state.diffusion_time,
                    forward_state.diffusion_mask,
                )
                loss = reconstruction.loss * (microbatch_cells / group_cells)

            # Preserve the production trainer's per-microbatch numerical guard
            # during both burn-in and recording. Burn-in still omits all expensive
            # module attribution, time-bin accounting, and parameter snapshots.
            if not bool(torch.isfinite(reconstruction.weighted_nll_sum).item()):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )
            if group_is_recorded:
                group_sufficient.append(
                    pack_sufficient_statistics(reconstruction)
                )
                bin_statistics, masks_per_cell, _positive_masks_per_cell = (
                    detached_time_bin_statistics(
                        distribution_parameters,
                        clean_expression,
                        forward_state.diffusion_time,
                        forward_state.diffusion_mask,
                        time_bins=args.time_bins,
                    )
                )
                group_time_bins.append(bin_statistics)
                group_times.append(forward_state.diffusion_time.detach())
                group_mask_counts.append(masks_per_cell)
            group_microbatches += 1
            scaler.scale(loss).backward()

            del denoiser_output
            del distribution_parameters
            del reconstruction
            del forward_state
            del clean_expression
            del loss

            should_step = (
                (batch_index + 1) % args.grad_accumulation_steps == 0
                or batch_index + 1 == num_batches
            )
            if not should_step:
                continue

            total_attempted_steps += 1
            if group_is_recorded:
                recorded_attempted_steps += 1
            else:
                burn_in_attempted_steps += 1
            scaler.unscale_(optimizer)
            if group_is_recorded:
                preclip = collect_preclip_group_tensors(groups)
                decoder_channel_squared, decoder_channel_max_abs = (
                    collect_decoder_channel_tensors(model)
                )
            returned_global_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
                # In FP16, unscale_ has already recorded non-finite gradients;
                # GradScaler.step must be allowed to skip that attempt. BF16 and
                # FP32 have no dynamic scaler, so fail immediately instead.
                error_if_nonfinite=not scaler.is_enabled(),
            )
            if group_is_recorded:
                snapshots = snapshot_trainable_parameters(groups)
                learning_rate_used = float(optimizer.param_groups[0]["lr"])
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_was_skipped = scaler.get_scale() < previous_scale

            if group_is_recorded:
                sufficient = torch.stack(group_sufficient).sum(dim=0)
                time_bin_totals = torch.stack(group_time_bins).sum(dim=0)
                times = torch.cat(group_times)
                mask_counts = torch.cat(group_mask_counts)
                if times.numel() != group_cells:
                    raise RuntimeError(
                        "Accumulated times do not match group cell count."
                    )
                time_features = time_feature_tensors(times, mask_counts)

            if optimizer_was_skipped:
                total_skipped_steps += 1
                if group_is_recorded:
                    recorded_skipped_steps += 1
                else:
                    burn_in_skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if group_is_recorded:
                    skip_record = {
                        "event": "optimizer_attempt_skipped",
                        "attempted_optimizer_step": total_attempted_steps,
                        "recorded_attempted_optimizer_step": (
                            recorded_attempted_steps
                        ),
                        "global_successful_optimizer_steps": (
                            total_successful_steps
                        ),
                        "successful_diagnostic_steps": len(successful_records),
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "group_cells": group_cells,
                        "reason": "FP16 dynamic-loss-scale overflow",
                    }
                    training.append_jsonl(metrics_path, skip_record)
                    print(json.dumps(skip_record, sort_keys=True), flush=True)
                    del snapshots
            else:
                scheduler.step()
                total_successful_steps += 1
                if group_is_recorded:
                    update_squared = collect_update_squared_norms(
                        groups, snapshots
                    )
                    optimizer.zero_grad(set_to_none=True)
                    step_payload = materialize_successful_step(
                        groups=groups,
                        preclip=preclip,
                        update_squared=update_squared,
                        decoder_channel_squared=decoder_channel_squared,
                        decoder_channel_max_abs=decoder_channel_max_abs,
                        returned_global_norm=returned_global_norm,
                        sufficient_statistics=sufficient,
                        time_bin_statistics=time_bin_totals,
                        time_features=time_features,
                        time_bins=args.time_bins,
                        max_grad_norm=args.max_grad_norm,
                    )
                    step_number = len(successful_records) + 1
                    expected_global_step = (
                        args.diagnostic_start_step + step_number
                    )
                    if total_successful_steps != expected_global_step:
                        raise RuntimeError(
                            "Recorded/global optimizer-step counters diverged."
                        )
                    step_payload.update(
                        {
                            "event": "gradient_diagnostic_step",
                            "diagnostic_step": step_number,
                            "global_optimizer_step": total_successful_steps,
                            "attempted_optimizer_step": total_attempted_steps,
                            "recorded_attempted_optimizer_step": (
                                recorded_attempted_steps
                            ),
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "microbatches": group_microbatches,
                            "group_cells": group_cells,
                            "learning_rate_used": learning_rate_used,
                            "learning_rate_next": float(
                                optimizer.param_groups[0]["lr"]
                            ),
                            "masked_fraction": step_payload["loss"][
                                "masked_count"
                            ]
                            / step_payload["loss"]["normalizer"],
                            "masked_positive_fraction": step_payload["loss"][
                                "masked_positive_count"
                            ]
                            / max(1, step_payload["loss"]["masked_count"]),
                            "step_elapsed_seconds": time.monotonic()
                            - group_started_at,
                            "peak_gpu_memory_bytes": (
                                torch.cuda.max_memory_allocated(device)
                            ),
                        }
                    )
                    successful_records.append(step_payload)
                    aggregate_time_bins += np.asarray(
                        [
                            [
                                record[field]
                                for record in step_payload["time_bins"]
                            ]
                            for field in SUFFICIENT_STATISTIC_FIELDS
                        ],
                        dtype=np.float64,
                    )
                    if step_number % args.diagnostic_every == 0:
                        training.append_jsonl(metrics_path, step_payload)
                    if step_number % args.log_every == 0 or step_number == 1:
                        console_record = {
                            "event": "gradient_diagnostic_progress",
                            "diagnostic_step": step_number,
                            "global_optimizer_step": total_successful_steps,
                            "target_recorded_steps": args.diagnostic_steps,
                            "target_total_steps": total_requested_steps,
                            "loss": step_payload["loss"]["loss"],
                            "preclip_global_grad_norm": step_payload[
                                "preclip_global_grad_norm"
                            ],
                            "clipped": step_payload["clipped"],
                            "mask_embedding_grad_norm": step_payload["modules"][
                                "absorbing_mask_embedding"
                            ]["preclip_gradient_norm"],
                            "lr_next": step_payload["learning_rate_next"],
                        }
                        print(
                            json.dumps(console_record, sort_keys=True),
                            flush=True,
                        )
                    del snapshots
                    del update_squared
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if total_successful_steps > args.diagnostic_start_step:
                        raise RuntimeError(
                            "Burn-in advanced past the diagnostic start boundary."
                        )
                    if (
                        total_successful_steps % args.log_every == 0
                        or total_successful_steps == 1
                        or total_successful_steps == args.diagnostic_start_step
                    ):
                        burn_in_progress = {
                            "event": "gradient_diagnostic_burn_in_progress",
                            "global_optimizer_step": total_successful_steps,
                            "burn_in_target_steps": args.diagnostic_start_step,
                            "target_total_steps": total_requested_steps,
                            "lr_next": float(optimizer.param_groups[0]["lr"]),
                        }
                        print(
                            json.dumps(burn_in_progress, sort_keys=True),
                            flush=True,
                        )
                    if total_successful_steps == args.diagnostic_start_step:
                        # Drain the burn-in stream exactly once so its queued GPU
                        # work and allocation peak cannot leak into recorded-window
                        # timing/memory. There is no per-step burn-in synchronization.
                        torch.cuda.synchronize(device)
                        burn_in_elapsed_seconds = time.monotonic() - start_time
                        burn_in_peak_gpu_memory_bytes = (
                            torch.cuda.max_memory_allocated(device)
                        )
                        recorded_start_time = time.monotonic()
                        torch.cuda.reset_peak_memory_stats(device)

            group_sufficient = []
            group_time_bins = []
            group_times = []
            group_mask_counts = []
            group_microbatches = 0

            if len(successful_records) >= args.diagnostic_steps:
                completion_reason = "target_optimizer_steps_reached"
                break
            if DIAGNOSTIC_STOP.requested:
                completion_reason = "signal"
                break

        if (
            len(successful_records) >= args.diagnostic_steps
            or DIAGNOSTIC_STOP.requested
        ):
            break

    end_time = time.monotonic()
    total_elapsed_seconds = end_time - start_time
    if burn_in_elapsed_seconds is None:
        burn_in_elapsed_seconds = total_elapsed_seconds
        burn_in_peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
    if recorded_start_time is None:
        recorded_elapsed_seconds = 0.0
        recorded_peak_gpu_memory_bytes = 0
    else:
        recorded_elapsed_seconds = end_time - recorded_start_time
        recorded_peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
    summary = build_summary(
        records=successful_records,
        groups=groups,
        aggregate_time_bins=aggregate_time_bins,
        recorded_attempted_steps=recorded_attempted_steps,
        recorded_skipped_steps=recorded_skipped_steps,
        target_recorded_steps=args.diagnostic_steps,
        diagnostic_start_step=args.diagnostic_start_step,
        total_successful_steps=total_successful_steps,
        total_attempted_steps=total_attempted_steps,
        total_skipped_steps=total_skipped_steps,
        burn_in_attempted_steps=burn_in_attempted_steps,
        burn_in_skipped_steps=burn_in_skipped_steps,
        recorded_elapsed_seconds=recorded_elapsed_seconds,
        burn_in_elapsed_seconds=burn_in_elapsed_seconds,
        total_elapsed_seconds=total_elapsed_seconds,
        recorded_peak_gpu_memory_bytes=recorded_peak_gpu_memory_bytes,
        burn_in_peak_gpu_memory_bytes=burn_in_peak_gpu_memory_bytes,
        completion_reason=completion_reason,
    )
    if DIAGNOSTIC_STOP.requested:
        summary["signal"] = DIAGNOSTIC_STOP.signal_number
    training.atomic_json_dump(summary, summary_path)
    training.append_jsonl(metrics_path, summary)
    print(json.dumps(training.jsonable(summary), sort_keys=True), flush=True)
    train_dataset.close()

    if DIAGNOSTIC_STOP.requested:
        return 130
    if (
        total_successful_steps != total_requested_steps
        or len(successful_records) != args.diagnostic_steps
    ):
        raise RuntimeError(
            "Scheduler horizon ended before the requested burn-in and recorded "
            "optimizer steps completed."
        )
    print(
        f"Gradient diagnostic completed: {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
