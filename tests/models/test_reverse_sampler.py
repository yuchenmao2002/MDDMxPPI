"""Small-shape tests for monotone absorbing reverse sampling."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from src.models.reverse_sampler import (
    ReverseSampler,
    SamplingConfig,
    linear_time_grid,
    sample_hurdle_distribution,
    sample_zero_truncated_normal,
)
from src.models.types import (
    DecoderOutput,
    HurdleDistributionParameters,
    ModelOutput,
)


class _RecordingDenoiser(nn.Module):
    """Deterministic parameter predictor with stochastic sampler readout."""

    def __init__(self, num_genes: int, *, positive_probability: float = 1.0) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.positive_probability = positive_probability
        self.calls = []

    def forward(
        self,
        expression_values: Tensor,
        diffusion_time: Tensor,
        diffusion_mask: Tensor,
        *,
        return_hidden_state: bool,
        output_hidden_states: bool,
        return_diagnostics: bool,
        compute_point_prediction: bool,
    ) -> ModelOutput:
        assert return_hidden_state is False
        assert output_hidden_states is False
        assert return_diagnostics is False
        assert compute_point_prediction is False
        self.calls.append(
            {
                "expression_values": expression_values.clone(),
                "diffusion_time": diffusion_time.clone(),
                "diffusion_mask": diffusion_mask.clone(),
            }
        )

        shape = (*diffusion_mask.shape, 1)
        if self.positive_probability == 1.0:
            logits = torch.full(
                shape,
                torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=expression_values.device,
            )
        elif self.positive_probability == 0.0:
            logits = torch.full(
                shape,
                -torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=expression_values.device,
            )
        else:
            probability = torch.tensor(
                self.positive_probability,
                dtype=torch.float32,
                device=expression_values.device,
            )
            logits = torch.logit(probability).expand(shape).clone()

        # Depend on t, not call count, so replaying a seed through the same
        # module remains deterministic.
        location = (1.0 + diffusion_time).view(-1, 1, 1).expand(shape).clone()
        scale = torch.full(
            shape,
            0.25,
            dtype=torch.float32,
            device=expression_values.device,
        )
        decoder_output = DecoderOutput(
            point_prediction=None,
            distribution_parameters=HurdleDistributionParameters(
                detection_logits=logits,
                positive_location=location,
                positive_scale=scale,
            ),
        )
        return ModelOutput(prediction=None, decoder_output=decoder_output)


def test_linear_time_grid_has_exact_endpoints_and_descends() -> None:
    grid = linear_time_grid(4)
    torch.testing.assert_close(
        grid,
        torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert grid.dtype == torch.float32
    assert bool((grid[:-1] > grid[1:]).all().item())


@pytest.mark.parametrize("bad_steps", [True, 0, -1, 1.5])
def test_sampling_config_rejects_invalid_step_count(bad_steps) -> None:
    with pytest.raises(ValueError, match="num_steps"):
        SamplingConfig(num_steps=bad_steps)


def test_sampling_config_rejects_unimplemented_schedule() -> None:
    with pytest.raises(ValueError, match="linear"):
        SamplingConfig(num_steps=2, schedule="cosine")


def test_hurdle_sampler_has_exact_zero_and_strictly_positive_branches() -> None:
    max_float = torch.finfo(torch.float32).max
    parameters = HurdleDistributionParameters(
        detection_logits=torch.tensor([[[-max_float], [max_float]]]),
        positive_location=torch.tensor([[[1.0], [-20.0]]]),
        positive_scale=torch.tensor([[[0.5], [0.1]]]),
    )
    result = sample_hurdle_distribution(
        parameters,
        generator=torch.Generator().manual_seed(7),
    )

    assert result.shape == (1, 2, 1)
    assert result.dtype == torch.float32
    assert result[0, 0, 0].item() == 0.0
    assert result[0, 1, 0].item() > 0.0
    assert bool(torch.isfinite(result).all().item())


def test_zero_truncated_normal_is_finite_positive_and_reproducible() -> None:
    location = torch.tensor([-1_000.0, -2.0, 0.0, 3.0, 1_000.0])
    scale = torch.tensor([0.1, 0.5, 1.0, 2.0, 0.1])
    first = sample_zero_truncated_normal(
        location,
        scale,
        generator=torch.Generator().manual_seed(19),
    )
    second = sample_zero_truncated_normal(
        location,
        scale,
        generator=torch.Generator().manual_seed(19),
    )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(first).all().item())
    assert bool((first > 0.0).all().item())


def test_standard_half_normal_sample_mean_matches_analytic_mean() -> None:
    num_samples = 50_000
    samples = sample_zero_truncated_normal(
        torch.zeros(num_samples),
        torch.ones(num_samples),
        generator=torch.Generator().manual_seed(71),
    )
    expected_mean = math.sqrt(2.0 / math.pi)

    assert samples.mean().item() == pytest.approx(expected_mean, abs=0.015)


def test_deep_negative_tail_preserves_sub_ulp_positive_excess() -> None:
    num_samples = 10_000
    samples = sample_zero_truncated_normal(
        torch.full((num_samples,), -1_000.0),
        torch.full((num_samples,), 0.1),
        generator=torch.Generator().manual_seed(73),
    )

    # For alpha=10,000 the conditional excess is asymptotically exponential
    # with expression-space mean scale/alpha = 1e-5.  Forming Z then subtracting
    # alpha in FP32 would round nearly every sample to zero.
    assert samples.mean().item() == pytest.approx(1.0e-5, rel=0.04)
    assert bool((samples > torch.finfo(torch.float32).tiny).any().item())


def test_one_step_sampling_reveals_every_gene() -> None:
    denoiser = _RecordingDenoiser(num_genes=7)
    sampler = ReverseSampler(denoiser, SamplingConfig(num_steps=1))
    output = sampler.sample(
        3,
        seed=11,
        return_diagnostics=True,
        return_trajectory=True,
    )

    assert len(denoiser.calls) == 1
    assert denoiser.calls[0]["diffusion_time"].dtype == torch.float32
    assert torch.equal(denoiser.calls[0]["diffusion_time"], torch.ones(3))
    assert bool(denoiser.calls[0]["diffusion_mask"].all().item())
    assert not bool(output.final_mask.any().item())
    assert bool((output.expression_values > 0.0).all().item())
    assert output.diagnostics is not None
    assert output.diagnostics.time_grid == (1.0, 0.0)
    assert output.diagnostics.masked_counts == (21, 0)
    assert output.diagnostics.revealed_counts == (21,)
    assert output.diagnostics.denoiser_calls == 1
    assert output.trajectory is not None and len(output.trajectory) == 2


def test_multistep_masks_are_monotone_and_visible_values_never_change() -> None:
    num_steps = 5
    denoiser = _RecordingDenoiser(num_genes=128)
    denoiser.train()
    sampler = ReverseSampler(denoiser, SamplingConfig(num_steps=num_steps))
    output = sampler.sample(
        2,
        seed=23,
        return_diagnostics=True,
        return_trajectory=True,
    )

    assert denoiser.training is True
    assert len(denoiser.calls) == num_steps
    expected_times = linear_time_grid(num_steps)[:-1]
    for call, expected_time in zip(denoiser.calls, expected_times):
        torch.testing.assert_close(
            call["diffusion_time"],
            expected_time.expand(2),
            rtol=0.0,
            atol=0.0,
        )
    assert output.trajectory is not None
    assert len(output.trajectory) == num_steps + 1
    assert bool(output.trajectory[0].diffusion_mask.all().item())
    assert torch.count_nonzero(output.trajectory[0].expression_values).item() == 0

    found_token_repredicted_on_next_step = False
    for step_index, (current, following) in enumerate(
        zip(output.trajectory[:-1], output.trajectory[1:])
    ):
        # A masked token can become visible, but a visible token cannot return
        # to MASK.  Values that were already visible are bitwise immutable.
        assert not bool((following.diffusion_mask & ~current.diffusion_mask).any())
        visible_before = ~current.diffusion_mask
        assert torch.equal(
            following.expression_values.squeeze(-1).masked_select(visible_before),
            current.expression_values.squeeze(-1).masked_select(visible_before),
        )
        assert torch.equal(
            denoiser.calls[step_index]["diffusion_mask"],
            current.diffusion_mask,
        )
        assert torch.equal(
            denoiser.calls[step_index]["expression_values"],
            current.expression_values,
        )
        if step_index + 1 < num_steps:
            still_masked = (
                denoiser.calls[step_index]["diffusion_mask"]
                & denoiser.calls[step_index + 1]["diffusion_mask"]
            )
            found_token_repredicted_on_next_step |= bool(still_masked.any().item())

    assert found_token_repredicted_on_next_step
    assert not bool(output.final_mask.any().item())
    assert bool(torch.isfinite(output.expression_values).all().item())
    assert bool((output.expression_values >= 0.0).all().item())
    assert output.diagnostics is not None
    assert all(
        later <= earlier
        for earlier, later in zip(
            output.diagnostics.masked_counts[:-1],
            output.diagnostics.masked_counts[1:],
        )
    )
    assert output.diagnostics.masked_counts[-1] == 0


def test_seed_replays_exactly_and_different_seed_changes_sample() -> None:
    sampler = ReverseSampler(
        _RecordingDenoiser(num_genes=32, positive_probability=0.65),
        SamplingConfig(num_steps=4),
    )
    first = sampler.sample(2, seed=101)
    second = sampler.sample(2, seed=101)
    different = sampler.sample(2, seed=102)

    assert torch.equal(first.expression_values, second.expression_values)
    assert torch.equal(first.final_mask, second.final_mask)
    assert not torch.equal(first.expression_values, different.expression_values)


def test_sample_accepts_generator_and_rejects_seed_with_generator() -> None:
    sampler = ReverseSampler(
        _RecordingDenoiser(num_genes=4),
        SamplingConfig(num_steps=2),
    )
    generator = torch.Generator().manual_seed(5)
    output = sampler.sample(1, generator=generator)
    assert output.expression_values.shape == (1, 4, 1)

    with pytest.raises(ValueError, match="either seed or generator"):
        sampler.sample(1, seed=5, generator=generator)
