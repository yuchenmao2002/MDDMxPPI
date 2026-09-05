from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from scripts import diagnose_masked_diffusion_gradients as diagnostic
from src.models.config import LossConfig, NUM_GENES
from src.models.losses import TimeWeightedHurdleNLLLoss
from src.models.types import HurdleDistributionParameters


class _ToyDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.absorbing_state_embedding = nn.Linear(1, 2, bias=False)
        self.decoder = nn.Linear(2, 3)
        self.gene_expression_encoder = nn.Linear(1, 2)
        self.gene_identity_encoder = nn.Module()
        self.gene_identity_encoder.projection = nn.Linear(2, 2, bias=False)
        self.backbone = nn.Linear(2, 2)


class _ToyTrainingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.denoiser = _ToyDenoiser()


def test_gradient_groups_cover_trainable_parameters_once() -> None:
    model = _ToyTrainingModule()
    groups = diagnostic.build_gradient_groups(model)

    assert {group.name for group in groups} == {
        "absorbing_mask_embedding",
        "decoder",
        "expression_encoder",
        "identity_projection",
        "performer_backbone",
    }
    assert sum(group.numel for group in groups) == sum(
        parameter.numel() for parameter in model.parameters()
    )

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    statistics = diagnostic.collect_preclip_group_tensors(groups)
    reconstructed_squared_norm = sum(
        value.gradient_squared_norm.item() for value in statistics.values()
    )
    expected_squared_norm = sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert reconstructed_squared_norm == pytest.approx(expected_squared_norm)


def test_detached_time_bins_reconstruct_batch_sufficient_statistics() -> None:
    target = torch.zeros((2, NUM_GENES, 1), dtype=torch.float32)
    target[0, 0, 0] = 1.0
    target[1, 100, 0] = 2.0
    times = torch.tensor((0.25, 0.75), dtype=torch.float32)
    mask = torch.zeros((2, NUM_GENES), dtype=torch.bool)
    mask[0, :100] = True
    mask[1, :300] = True

    parameters = HurdleDistributionParameters(
        detection_logits=torch.zeros_like(target),
        positive_location=torch.zeros_like(target),
        positive_scale=torch.ones_like(target),
    )
    bin_statistics, masks_per_cell, positive_per_cell = (
        diagnostic.detached_time_bin_statistics(
            parameters,
            target,
            times,
            mask,
            time_bins=4,
        )
    )

    assert bin_statistics.shape == (8, 4)
    assert masks_per_cell.tolist() == [100, 300]
    assert positive_per_cell.tolist() == [1, 1]
    assert int(bin_statistics[4].sum().item()) == 2
    assert int(bin_statistics[5].sum().item()) == 400
    assert int(bin_statistics[6].sum().item()) == 398
    assert int(bin_statistics[7].sum().item()) == 2
    assert torch.allclose(
        bin_statistics[0],
        bin_statistics[2] + bin_statistics[3],
    )

    production = TimeWeightedHurdleNLLLoss(LossConfig())(
        parameters,
        target,
        times,
        mask,
    )
    for row, field in enumerate(diagnostic.SUFFICIENT_STATISTIC_FIELDS):
        expected = getattr(production, field).detach().double()
        assert bin_statistics[row].sum() == pytest.approx(
            expected,
            rel=2e-6,
            abs=1e-5,
        )


def test_materialized_step_reconstructs_disjoint_global_norm() -> None:
    model = _ToyTrainingModule()
    groups = diagnostic.build_gradient_groups(model)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)

    preclip = diagnostic.collect_preclip_group_tensors(groups)
    returned_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    snapshots = diagnostic.snapshot_trainable_parameters(groups)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(parameter.grad, alpha=-0.1)
    update_squared = diagnostic.collect_update_squared_norms(groups, snapshots)

    sufficient = torch.tensor(
        (10.0, 100.0, 4.0, 6.0, 2.0, 20.0, 18.0, 2.0),
        dtype=torch.float64,
    )
    time_bins = torch.zeros((8, 4), dtype=torch.float64)
    time_bins[:, 0] = sufficient
    time_features = torch.zeros(
        len(diagnostic.TIME_QUANTILE_FIELDS), dtype=torch.float32
    )
    payload = diagnostic.materialize_successful_step(
        groups=groups,
        preclip=preclip,
        update_squared=update_squared,
        decoder_channel_squared=torch.zeros(3),
        decoder_channel_max_abs=torch.zeros(3),
        returned_global_norm=returned_norm,
        sufficient_statistics=sufficient,
        time_bin_statistics=time_bins,
        time_features=time_features,
        time_bins=4,
        max_grad_norm=1.0,
    )

    assert payload["preclip_global_grad_norm"] == pytest.approx(
        payload["reconstructed_preclip_global_grad_norm"], rel=1e-6
    )
    assert sum(
        module["gradient_energy_fraction"]
        for module in payload["modules"].values()
    ) == pytest.approx(1.0)
    assert payload["time_bin_relative_nll_accounting_error"] == 0.0


def test_describe_and_constant_correlation_contracts() -> None:
    summary = diagnostic.describe([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p50"] == pytest.approx(2.5)
    assert diagnostic.correlation([1.0, 1.0], [2.0, 3.0]) is None
    assert diagnostic.correlation([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)


def test_scheduler_step_500_matches_baseline_horizon() -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = AdamW((parameter,), lr=2e-4)
    scheduler = diagnostic.training.build_scheduler(
        optimizer,
        total_steps=20_780,
        warmup_ratio=0.05,
        min_lr_ratio=0.1,
    )
    for _ in range(500):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        9.643888354186719e-05,
        rel=1e-14,
    )


def test_diagnostic_start_step_defaults_to_zero_and_rejects_negative(
    tmp_path,
) -> None:
    args = diagnostic.build_parser().parse_args(
        ["--output-dir", str(tmp_path / "diagnostic")]
    )
    assert args.diagnostic_start_step == 0
    diagnostic.validate_diagnostic_args(args)

    args.diagnostic_start_step = -1
    with pytest.raises(ValueError, match="diagnostic-start-step"):
        diagnostic.validate_diagnostic_args(args)


def test_post_warmup_summary_separates_window_burn_in_and_totals() -> None:
    parameter = nn.Parameter(torch.ones(1))
    group = diagnostic.GradientGroup(
        name="toy",
        parameters=(("toy.weight", parameter),),
    )

    def record(global_step: int, *, clipped: bool) -> dict[str, object]:
        return {
            "global_optimizer_step": global_step,
            "preclip_global_grad_norm": 2.0,
            "clipped": clipped,
            "group_cells": 256,
            "loss": {"loss": 0.25, "positive_loss": 0.15},
            "time": {
                "min": 0.01,
                "max_inverse_time_with_mask": 100.0,
                "inverse_weighted_mask_mass": 1.0,
            },
            "modules": {
                "toy": {
                    "preclip_gradient_norm": 2.0,
                    "update_to_parameter_ratio": 1e-4,
                }
            },
            "decoder_channels": {
                name: {"preclip_gradient_norm": 0.5}
                for name in diagnostic.DECODER_CHANNEL_NAMES
            },
        }

    summary = diagnostic.build_summary(
        records=[record(1201, clipped=True), record(1202, clipped=False)],
        groups=(group,),
        aggregate_time_bins=torch.zeros(
            (len(diagnostic.SUFFICIENT_STATISTIC_FIELDS), 16),
            dtype=torch.float64,
        ).numpy(),
        recorded_attempted_steps=3,
        recorded_skipped_steps=1,
        target_recorded_steps=2,
        diagnostic_start_step=1200,
        total_successful_steps=1202,
        total_attempted_steps=1203,
        total_skipped_steps=1,
        burn_in_attempted_steps=1200,
        burn_in_skipped_steps=0,
        recorded_elapsed_seconds=20.0,
        burn_in_elapsed_seconds=100.0,
        total_elapsed_seconds=120.0,
        recorded_peak_gpu_memory_bytes=200,
        burn_in_peak_gpu_memory_bytes=100,
        completion_reason="target_optimizer_steps_reached",
    )

    # Legacy counters remain scoped to the recorded diagnostic window.
    assert summary["target_optimizer_steps"] == 2
    assert summary["successful_optimizer_steps"] == 2
    assert summary["attempted_optimizer_steps"] == 3
    assert summary["skipped_optimizer_steps"] == 1
    assert summary["clip_count"] == 1
    assert summary["clip_rate"] == pytest.approx(0.5)

    window = summary["recorded_window"]
    assert window["first_target_global_optimizer_step"] == 1201
    assert window["last_target_global_optimizer_step"] == 1202
    assert window["first_recorded_global_optimizer_step"] == 1201
    assert window["last_recorded_global_optimizer_step"] == 1202
    assert summary["burn_in"]["successful_optimizer_steps"] == 1200
    assert summary["total_execution"]["successful_optimizer_steps"] == 1202
    assert summary["total_execution"]["target_successful_optimizer_steps"] == 1202
