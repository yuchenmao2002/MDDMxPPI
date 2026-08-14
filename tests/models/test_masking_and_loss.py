"""Contracts for absorbing corruption, MASK embedding, and hurdle NLL."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from src.models.config import ForwardProcessConfig, LossConfig, NUM_GENES
from src.models.losses import TimeWeightedHurdleNLLLoss
from src.models.masking import AbsorbingMaskForwardProcess, AbsorbingStateEmbedding
from src.models.types import HurdleDistributionParameters


def _parameters(
    batch_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> HurdleDistributionParameters:
    shape = (batch_size, NUM_GENES, 1)
    return HurdleDistributionParameters(
        detection_logits=torch.zeros(shape, dtype=dtype, requires_grad=requires_grad),
        positive_location=torch.ones(shape, dtype=dtype, requires_grad=requires_grad),
        positive_scale=torch.ones(shape, dtype=dtype, requires_grad=requires_grad),
    )


def _truncated_normal_nll(x: float, mu: float, sigma: float) -> float:
    x_tensor = torch.tensor(x, dtype=torch.float64)
    mu_tensor = torch.tensor(mu, dtype=torch.float64)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float64)
    return float(
        (
            0.5 * ((x_tensor - mu_tensor) / sigma_tensor).square()
            + torch.log(sigma_tensor)
            + 0.5 * math.log(2.0 * math.pi)
            + torch.special.log_ndtr(mu_tensor / sigma_tensor)
        ).item()
    )


def test_sample_times_are_independent_fp32_uniform_draws() -> None:
    process = AbsorbingMaskForwardProcess(ForwardProcessConfig())
    generator = torch.Generator(device="cpu").manual_seed(17)

    diffusion_time = process.sample_times(
        32,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert diffusion_time.shape == (32,)
    assert diffusion_time.dtype == torch.float32
    assert diffusion_time.device.type == "cpu"
    assert torch.all(diffusion_time >= 0.0)
    assert torch.all(diffusion_time < 1.0)
    assert torch.unique(diffusion_time).numel() > 1


def test_explicit_endpoint_times_produce_none_and_all_masked_rows() -> None:
    process = AbsorbingMaskForwardProcess(ForwardProcessConfig())
    diffusion_time = torch.tensor([0.0, 1.0], dtype=torch.float32)

    output = process.sample(
        2,
        device=torch.device("cpu"),
        diffusion_time=diffusion_time,
        generator=torch.Generator(device="cpu").manual_seed(3),
    )

    assert output.diffusion_time is diffusion_time
    assert output.diffusion_mask.shape == (2, NUM_GENES)
    assert output.diffusion_mask.dtype == torch.bool
    assert not output.diffusion_mask[0].any()
    assert output.diffusion_mask[1].all()


def test_gene_masks_are_separate_bernoulli_draws() -> None:
    process = AbsorbingMaskForwardProcess(ForwardProcessConfig())
    diffusion_time = torch.full((2,), 0.5, dtype=torch.float32)

    diffusion_mask = process.sample_mask(
        diffusion_time,
        generator=torch.Generator(device="cpu").manual_seed(11),
    )

    row_fractions = diffusion_mask.float().mean(dim=1)
    assert torch.all((row_fractions > 0.45) & (row_fractions < 0.55))
    assert not torch.equal(diffusion_mask[0], diffusion_mask[1])
    assert diffusion_mask[0].any() and (~diffusion_mask[0]).any()


def test_mask_embedding_is_shared_and_cuts_clean_value_gradient() -> None:
    module = AbsorbingStateEmbedding(d_model=4)
    encoded = torch.randn(1, NUM_GENES, 4, requires_grad=True)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)
    diffusion_mask[0, 5] = True
    diffusion_mask[0, 19] = True

    masked = module(encoded, diffusion_mask)

    assert masked.shape == encoded.shape
    assert masked.dtype == encoded.dtype
    assert masked.device == encoded.device
    torch.testing.assert_close(masked[0, 5], module.mask_embedding)
    torch.testing.assert_close(masked[0, 19], module.mask_embedding)
    torch.testing.assert_close(masked[0, 0], encoded[0, 0])

    masked.sum().backward()
    assert encoded.grad is not None
    assert torch.count_nonzero(encoded.grad[0, 5]) == 0
    assert torch.count_nonzero(encoded.grad[0, 19]) == 0
    torch.testing.assert_close(encoded.grad[0, 0], torch.ones(4))
    torch.testing.assert_close(module.mask_embedding.grad, torch.full((4,), 2.0))


def test_mask_embedding_uses_caller_rng_and_normal_initialization() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(123)
        first = AbsorbingStateEmbedding(d_model=512)
        torch.manual_seed(123)
        second = AbsorbingStateEmbedding(d_model=512)

    torch.testing.assert_close(first.mask_embedding, second.mask_embedding)
    assert first.mask_embedding.detach().std().item() == pytest.approx(0.02, abs=0.003)


def test_hurdle_nll_matches_manual_zero_and_positive_branches() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1)
    target = torch.zeros(1, NUM_GENES, 1)
    diffusion_time = torch.tensor([0.5], dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)

    with torch.no_grad():
        parameters.detection_logits[0, 0, 0] = math.log(3.0)
        parameters.detection_logits[0, 1, 0] = -math.log(2.0)
        parameters.positive_location[0, 1, 0] = 1.25
        parameters.positive_scale[0, 1, 0] = 0.75
        target[0, 1, 0] = 2.0
    diffusion_mask[0, :2] = True

    output = criterion(parameters, target, diffusion_time, diffusion_mask)

    zero_nll = F.softplus(torch.tensor(math.log(3.0))).item()
    positive_gate_nll = F.softplus(torch.tensor(math.log(2.0))).item()
    positive_value_nll = _truncated_normal_nll(2.0, 1.25, 0.75)
    expected_zero_sum = 2.0 * zero_nll
    expected_positive_sum = 2.0 * (positive_gate_nll + positive_value_nll)

    assert output.weighted_zero_nll_sum.item() == pytest.approx(expected_zero_sum)
    assert output.weighted_positive_nll_sum.item() == pytest.approx(
        expected_positive_sum
    )
    assert output.weighted_nll_sum.item() == pytest.approx(
        expected_zero_sum + expected_positive_sum
    )
    assert output.loss.item() == pytest.approx(
        (expected_zero_sum + expected_positive_sum) / NUM_GENES
    )
    assert output.normalizer.dtype == torch.int64
    assert output.normalizer.item() == NUM_GENES
    assert output.cell_count.item() == 1
    assert output.masked_count.item() == 2
    assert output.masked_zero_count.item() == 1
    assert output.masked_positive_count.item() == 1


def test_loss_uses_inverse_time_and_fixed_cell_gene_normalizer() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(2)
    target = torch.zeros(2, NUM_GENES, 1)
    diffusion_time = torch.tensor([0.25, 0.5], dtype=torch.float32)
    diffusion_mask = torch.zeros(2, NUM_GENES, dtype=torch.bool)
    diffusion_mask[0, 0] = True
    diffusion_mask[1, :3] = True

    output = criterion(parameters, target, diffusion_time, diffusion_mask)

    token_nll = math.log(2.0)
    expected = (1.0 / 0.25 + 3.0 / 0.5) * token_nll
    assert output.weighted_nll_sum.item() == pytest.approx(expected)
    assert output.loss.item() == pytest.approx(expected / (2 * NUM_GENES))
    assert output.normalizer.item() == 2 * NUM_GENES
    assert output.masked_count.item() == 4


def test_equal_per_cell_losses_are_not_reweighted_by_mask_count() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(2)
    target = torch.zeros(2, NUM_GENES, 1)
    diffusion_time = torch.tensor([0.25, 0.75], dtype=torch.float32)
    diffusion_mask = torch.zeros(2, NUM_GENES, dtype=torch.bool)
    # Choose mask counts proportional to t.  Each row then has the same
    # importance-corrected NLL contribution despite very different mask counts.
    diffusion_mask[0, :1] = True
    diffusion_mask[1, :3] = True

    combined = criterion(parameters, target, diffusion_time, diffusion_mask)
    first = criterion(
        HurdleDistributionParameters(
            detection_logits=parameters.detection_logits[:1],
            positive_location=parameters.positive_location[:1],
            positive_scale=parameters.positive_scale[:1],
        ),
        target[:1],
        diffusion_time[:1],
        diffusion_mask[:1],
    )
    second = criterion(
        HurdleDistributionParameters(
            detection_logits=parameters.detection_logits[1:],
            positive_location=parameters.positive_location[1:],
            positive_scale=parameters.positive_scale[1:],
        ),
        target[1:],
        diffusion_time[1:],
        diffusion_mask[1:],
    )

    assert first.weighted_nll_sum.item() == pytest.approx(
        second.weighted_nll_sum.item()
    )
    assert combined.loss.item() == pytest.approx(
        0.5 * (first.loss.item() + second.loss.item())
    )


def test_unmasked_t_zero_row_contributes_differentiable_zero() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1, requires_grad=True)
    target = torch.zeros(1, NUM_GENES, 1)
    diffusion_time = torch.zeros(1, dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)

    output = criterion(parameters, target, diffusion_time, diffusion_mask)

    assert output.loss.dtype == torch.float32
    assert output.loss.item() == 0.0
    assert output.weighted_nll_sum.item() == 0.0
    assert output.loss.requires_grad
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for tensor in (
        parameters.detection_logits,
        parameters.positive_location,
        parameters.positive_scale,
    ):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


def test_masked_t_zero_row_is_rejected() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1)
    target = torch.zeros(1, NUM_GENES, 1)
    diffusion_time = torch.zeros(1, dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)
    diffusion_mask[0, 0] = True

    with pytest.raises(ValueError, match="diffusion_time=0"):
        criterion(parameters, target, diffusion_time, diffusion_mask)


def test_hurdle_nll_converts_low_precision_inputs_before_probability_math() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, NUM_GENES, 1, dtype=torch.float16)
    target[0, 7, 0] = 1.5
    diffusion_time = torch.tensor([0.5], dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)
    diffusion_mask[0, 7] = True

    output = criterion(parameters, target, diffusion_time, diffusion_mask)

    assert output.loss.dtype == torch.float32
    assert output.weighted_nll_sum.dtype == torch.float32
    assert output.weighted_positive_nll_sum.dtype == torch.float32
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert parameters.detection_logits.grad is not None
    assert parameters.positive_location.grad is not None
    assert parameters.positive_scale.grad is not None


def test_extreme_negative_location_has_stable_value_and_finite_gradients() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1, requires_grad=True)
    target = torch.zeros(1, NUM_GENES, 1)
    diffusion_time = torch.ones(1, dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)

    # At z=mu/sigma=-1e6, the direct formula subtracts two values near 5e11.
    # The first omitted tail-asymptotic correction is O(z^-2), so this
    # independent approximation is far more accurate than FP32 resolution.
    extreme_location = -1_000_000.0
    with torch.no_grad():
        parameters.positive_location[0, 3, 0] = extreme_location
        target[0, 3, 0] = 1.0
    diffusion_mask[0, 3] = True

    output = criterion(parameters, target, diffusion_time, diffusion_mask)

    expected_value_nll = (
        0.5 - extreme_location - math.log(-extreme_location)
    )
    expected_token_nll = math.log(2.0) + expected_value_nll
    assert torch.isfinite(output.weighted_positive_nll_sum)
    assert output.weighted_positive_nll_sum.item() == pytest.approx(
        expected_token_nll,
        abs=0.25,
    )

    output.loss.backward()
    for tensor in (
        parameters.detection_logits,
        parameters.positive_location,
        parameters.positive_scale,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert parameters.positive_location.grad[0, 3, 0] != 0
    assert parameters.positive_scale.grad[0, 3, 0] != 0


def test_only_masked_branch_parameters_receive_nonzero_gradients() -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1, requires_grad=True)
    target = torch.zeros(1, NUM_GENES, 1)
    target[0, 1, 0] = 2.0
    diffusion_time = torch.tensor([0.5], dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)
    diffusion_mask[0, :2] = True

    output = criterion(parameters, target, diffusion_time, diffusion_mask)
    output.loss.backward()

    assert parameters.detection_logits.grad is not None
    assert parameters.positive_location.grad is not None
    assert parameters.positive_scale.grad is not None
    assert parameters.detection_logits.grad[0, 0, 0] != 0
    assert parameters.detection_logits.grad[0, 1, 0] != 0
    # Zero targets train only the hurdle gate, not positive-distribution params.
    assert parameters.positive_location.grad[0, 0, 0] == 0
    assert parameters.positive_scale.grad[0, 0, 0] == 0
    assert parameters.positive_location.grad[0, 1, 0] != 0
    assert parameters.positive_scale.grad[0, 1, 0] != 0
    assert torch.count_nonzero(parameters.detection_logits.grad[:, 2:, :]) == 0
    assert torch.count_nonzero(parameters.positive_location.grad[:, 2:, :]) == 0
    assert torch.count_nonzero(parameters.positive_scale.grad[:, 2:, :]) == 0


@pytest.mark.parametrize(
    ("mutation", "expected_exception"),
    [
        ("negative_target", ValueError),
        ("nonfinite_target", ValueError),
        ("nonpositive_scale", ValueError),
        ("nonfinite_logit", ValueError),
        ("wrong_parameter_shape", ValueError),
        ("integer_parameter", TypeError),
        ("non_fp32_time", TypeError),
        ("non_boolean_mask", TypeError),
    ],
)
def test_hurdle_nll_rejects_invalid_inputs(
    mutation: str,
    expected_exception: type[Exception],
) -> None:
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    parameters = _parameters(1)
    target = torch.zeros(1, NUM_GENES, 1)
    diffusion_time = torch.tensor([0.5], dtype=torch.float32)
    diffusion_mask = torch.zeros(1, NUM_GENES, dtype=torch.bool)

    if mutation == "negative_target":
        target[0, 0, 0] = -1.0
    elif mutation == "nonfinite_target":
        target[0, 0, 0] = float("nan")
    elif mutation == "nonpositive_scale":
        parameters.positive_scale[0, 0, 0] = 0.0
    elif mutation == "nonfinite_logit":
        parameters.detection_logits[0, 0, 0] = float("inf")
    elif mutation == "wrong_parameter_shape":
        parameters = HurdleDistributionParameters(
            detection_logits=torch.zeros(1, NUM_GENES),
            positive_location=parameters.positive_location,
            positive_scale=parameters.positive_scale,
        )
    elif mutation == "integer_parameter":
        parameters = HurdleDistributionParameters(
            detection_logits=torch.zeros(1, NUM_GENES, 1, dtype=torch.int64),
            positive_location=parameters.positive_location,
            positive_scale=parameters.positive_scale,
        )
    elif mutation == "non_fp32_time":
        diffusion_time = diffusion_time.to(torch.float16)
    elif mutation == "non_boolean_mask":
        diffusion_mask = diffusion_mask.float()
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(f"Unhandled mutation: {mutation}")

    with pytest.raises(expected_exception):
        criterion(parameters, target, diffusion_time, diffusion_mask)
