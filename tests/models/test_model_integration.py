"""Small-shape integration tests for the top-level deterministic/training APIs."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import src.models.masked_diffusion_training as model_module
import src.models.masked_expression_denoiser as denoiser_module
from src.models.config import LossConfig
from src.models.losses import TimeWeightedHurdleNLLLoss
from src.models.masked_diffusion_training import (
    MaskedDiffusionTrainingModule,
    MaskedExpressionDenoiser as LegacyMaskedExpressionDenoiser,
)
from src.models.masked_expression_denoiser import MaskedExpressionDenoiser
from src.models.types import (
    BackboneOutput,
    DecoderOutput,
    ForwardProcessOutput,
    HurdleDistributionParameters,
    TimeWeightedHurdleNLLOutput,
)


class _IdentityEncoder(nn.Module):
    def __init__(self, num_genes: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_genes, d_model))

    def forward(self) -> Tensor:
        return self.weight


class _PointwiseExpressionEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, values: Tensor) -> Tensor:
        return values * self.scale.view(1, 1, -1)


class _AbsorbingState(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mask_embedding = nn.Parameter(torch.full((d_model,), 0.25))

    def forward(self, encoded: Tensor, mask: Tensor) -> Tensor:
        return torch.where(
            mask.unsqueeze(-1),
            self.mask_embedding.view(1, 1, -1),
            encoded,
        )


class _IdentityBackbone(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        context,
        *,
        output_hidden_states: bool = False,
        return_diagnostics: bool = False,
    ) -> BackboneOutput:
        del context, return_diagnostics
        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )


class _SmallHurdleDecoder(nn.Module):
    """Small differentiable hurdle head used only to exercise model wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.point_prediction_requests = []

    def forward(
        self,
        hidden_states: Tensor,
        *,
        compute_point_prediction: bool = True,
    ) -> DecoderOutput:
        self.point_prediction_requests.append(compute_point_prediction)
        location = hidden_states[..., :1].float()
        logits = hidden_states[..., 1:2].float()
        scale = 0.1 + F.softplus(hidden_states[..., 2:3].float())
        # This fake point readout need only preserve the public expectation
        # contract; distribution math itself is covered by loss/decoder tests.
        point_prediction = (
            torch.sigmoid(logits) * F.softplus(location)
            if compute_point_prediction
            else None
        )
        return DecoderOutput(
            point_prediction=point_prediction,
            distribution_parameters=HurdleDistributionParameters(
                detection_logits=logits,
                positive_location=location,
                positive_scale=scale,
            ),
        )


class _RecordingHurdleLoss(nn.Module):
    """Delegate to the real loss while preserving the exact received inputs."""

    def __init__(self, criterion: TimeWeightedHurdleNLLLoss) -> None:
        super().__init__()
        self.criterion = criterion
        self.received_parameters = None
        self.received_time = None
        self.received_mask = None

    def forward(
        self,
        parameters: HurdleDistributionParameters,
        target: Tensor,
        diffusion_time: Tensor,
        diffusion_mask: Tensor,
    ) -> TimeWeightedHurdleNLLOutput:
        self.received_parameters = parameters
        self.received_time = diffusion_time
        self.received_mask = diffusion_mask
        return self.criterion(
            parameters,
            target,
            diffusion_time,
            diffusion_mask,
        )


class _RecordingForwardProcess:
    def __init__(self, num_genes: int) -> None:
        self.num_genes = num_genes
        self.calls = 0

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        diffusion_time=None,
        generator=None,
    ) -> ForwardProcessOutput:
        del generator
        self.calls += 1
        if diffusion_time is None:
            diffusion_time = torch.full(
                (batch_size,), 0.5, dtype=torch.float32, device=device
            )
        draws = torch.arange(self.num_genes, device=device).unsqueeze(0)
        thresholds = diffusion_time.unsqueeze(1) * self.num_genes
        return ForwardProcessOutput(
            diffusion_time=diffusion_time,
            diffusion_mask=draws < thresholds,
        )


def _small_denoiser(monkeypatch, *, num_genes: int = 5, d_model: int = 4):
    monkeypatch.setattr(denoiser_module, "NUM_GENES", num_genes)
    monkeypatch.setattr(denoiser_module, "DEFAULT_D_MODEL", d_model)
    return MaskedExpressionDenoiser(
        gene_identity_encoder=_IdentityEncoder(num_genes, d_model),
        gene_expression_encoder=_PointwiseExpressionEncoder(d_model),
        absorbing_state_embedding=_AbsorbingState(d_model),
        backbone=_IdentityBackbone(),
        decoder=_SmallHurdleDecoder(),
    )


def test_legacy_module_reexports_the_split_denoiser_class() -> None:
    assert LegacyMaskedExpressionDenoiser is MaskedExpressionDenoiser
    assert model_module.MaskedExpressionDenoiser is MaskedExpressionDenoiser


def test_masked_placeholders_are_sanitized_and_decoder_output_is_preserved(
    monkeypatch,
) -> None:
    denoiser = _small_denoiser(monkeypatch)
    mask = torch.tensor([[False, True, False, True, False]])
    time = torch.tensor([0.4], dtype=torch.float32)
    first = torch.tensor([[[1.0], [-1.0e30], [2.0], [7.0], [3.0]]])
    second = first.clone()
    second[0, 1, 0] = 1.0e30
    second[0, 3, 0] = -9.0e20

    first_output = denoiser(first, time, mask)
    second_output = denoiser(second, time, mask)

    assert first_output.prediction is not None
    assert second_output.prediction is not None
    torch.testing.assert_close(first_output.prediction, second_output.prediction)
    assert first_output.decoder_output is not None
    assert second_output.decoder_output is not None
    assert first_output.prediction is first_output.decoder_output.point_prediction
    first_parameters = first_output.decoder_output.distribution_parameters
    second_parameters = second_output.decoder_output.distribution_parameters
    assert first_parameters is not None
    assert second_parameters is not None
    torch.testing.assert_close(
        first_parameters.detection_logits,
        second_parameters.detection_logits,
    )
    torch.testing.assert_close(
        first_parameters.positive_location,
        second_parameters.positive_location,
    )
    torch.testing.assert_close(
        first_parameters.positive_scale,
        second_parameters.positive_scale,
    )


def test_masked_input_values_have_no_gradient_path(monkeypatch) -> None:
    denoiser = _small_denoiser(monkeypatch)
    expression = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    expression.requires_grad_()
    mask = torch.tensor([[False, True, False, True, False]])

    output = denoiser(
        expression,
        torch.tensor([0.5], dtype=torch.float32),
        mask,
    )
    assert output.prediction is not None
    output.prediction.sum().backward()

    assert expression.grad is not None
    assert expression.grad[0, 1, 0].item() == 0.0
    assert expression.grad[0, 3, 0].item() == 0.0
    assert torch.all(expression.grad[0, ~mask[0], 0] != 0)


def test_training_wrapper_state_supply_rules_and_zero_mask(monkeypatch) -> None:
    num_genes = 5
    denoiser = _small_denoiser(monkeypatch, num_genes=num_genes)
    process = _RecordingForwardProcess(num_genes)
    criterion = _RecordingHurdleLoss(TimeWeightedHurdleNLLLoss(LossConfig()))
    model = MaskedDiffusionTrainingModule(
        denoiser=denoiser,
        forward_process=process,
        reconstruction_loss=criterion,
    )
    clean = torch.arange(num_genes, dtype=torch.float32).view(1, num_genes, 1)

    supplied_time = torch.tensor([0.4], dtype=torch.float32)
    time_only = model(
        clean,
        diffusion_time=supplied_time,
    )
    assert time_only.prediction is None
    assert denoiser.decoder.point_prediction_requests[-1] is False
    assert process.calls == 1
    assert time_only.diffusion_mask.any()
    assert criterion.received_time is supplied_time
    assert criterion.received_mask is time_only.diffusion_mask
    assert criterion.received_parameters is not None
    assert time_only.cell_count.item() == 1
    assert time_only.normalizer.item() == num_genes
    assert time_only.masked_count.item() == 2
    assert time_only.masked_zero_count.item() == 1
    assert time_only.masked_positive_count.item() == 1
    torch.testing.assert_close(
        time_only.weighted_nll_sum,
        time_only.weighted_zero_nll_sum
        + time_only.weighted_positive_nll_sum,
    )
    torch.testing.assert_close(
        time_only.loss,
        time_only.weighted_nll_sum / num_genes,
    )
    assert time_only.loss is time_only.reconstruction_loss

    with pytest.raises(ValueError, match="diffusion_time is required"):
        model(clean, diffusion_mask=torch.zeros(1, num_genes, dtype=torch.bool))

    explicit_time = torch.zeros(1, dtype=torch.float32)
    explicit_mask = torch.zeros(1, num_genes, dtype=torch.bool)
    zero_mask = model(
        clean,
        diffusion_time=explicit_time,
        diffusion_mask=explicit_mask,
    )
    assert process.calls == 1  # Explicit state was not resampled.
    assert criterion.received_time is explicit_time
    assert criterion.received_mask is explicit_mask
    assert zero_mask.masked_count.item() == 0
    assert zero_mask.masked_zero_count.item() == 0
    assert zero_mask.masked_positive_count.item() == 0
    assert zero_mask.weighted_nll_sum.item() == 0.0
    assert zero_mask.weighted_zero_nll_sum.item() == 0.0
    assert zero_mask.weighted_positive_nll_sum.item() == 0.0
    assert zero_mask.normalizer.item() == num_genes
    assert zero_mask.loss.item() == 0.0
    assert zero_mask.loss.requires_grad
    zero_mask.loss.backward()
    assert denoiser.gene_identity_encoder.weight.grad is not None
    assert torch.count_nonzero(denoiser.gene_identity_encoder.weight.grad) == 0

    full_mask = model(
        clean,
        diffusion_time=torch.ones(1, dtype=torch.float32),
        diffusion_mask=torch.ones(1, num_genes, dtype=torch.bool),
    )
    assert full_mask.masked_count.item() == num_genes
    assert full_mask.normalizer.item() == num_genes
    assert torch.isfinite(full_mask.loss)

    with pytest.raises(ValueError, match="Rows at t=0"):
        model(
            clean,
            diffusion_time=torch.zeros(1, dtype=torch.float32),
            diffusion_mask=torch.ones(1, num_genes, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="Rows at t=1"):
        model(
            clean,
            diffusion_time=torch.ones(1, dtype=torch.float32),
            diffusion_mask=torch.zeros(1, num_genes, dtype=torch.bool),
        )


def test_denoiser_can_explicitly_skip_point_prediction(monkeypatch) -> None:
    denoiser = _small_denoiser(monkeypatch)
    output = denoiser(
        torch.ones((1, 5, 1), dtype=torch.float32),
        torch.full((1,), 0.5, dtype=torch.float32),
        torch.tensor([[False, True, False, True, False]]),
        compute_point_prediction=False,
    )

    assert output.prediction is None
    assert output.decoder_output.point_prediction is None
    assert denoiser.decoder.point_prediction_requests[-1] is False
