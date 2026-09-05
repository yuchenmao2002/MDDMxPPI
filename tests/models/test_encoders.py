"""Contract tests for the fixed-axis model input and output heads."""

import math
from types import SimpleNamespace

import pytest
import torch

import src.models.gene_expression_decoder as decoder_module
import src.models.gene_expression_encoder as expression_module
from src.models.config import DecoderConfig, GeneExpressionEncoderConfig, LossConfig
from src.models.gene_expression_decoder import GeneExpressionDecoder
from src.models.gene_expression_encoder import GeneExpressionEncoder
from src.models.gene_identity_encoder import GeneIdentityEncoder
from src.models.losses import TimeWeightedHurdleNLLLoss


def _small_identity_config(seed: int = 23) -> SimpleNamespace:
    return SimpleNamespace(
        num_genes=5,
        source_dim=7,
        d_model=3,
        trainable=True,
        projection_bias=False,
        projection_seed=seed,
    )


def test_identity_encoder_copies_asset_and_does_not_pollute_global_rng() -> None:
    config = _small_identity_config()
    initial_weight = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    torch.manual_seed(101)
    rng_before = torch.get_rng_state().clone()

    encoder = GeneIdentityEncoder(config, initial_weight)

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert encoder.embedding.weight.data_ptr() != initial_weight.data_ptr()
    assert torch.equal(encoder.embedding.weight, initial_weight)
    assert encoder.embedding.weight.requires_grad
    assert encoder.projection.bias is None
    assert encoder.projection.weight.requires_grad
    assert encoder().shape == (5, 3)

    identity = torch.eye(3)
    row_gram = encoder.projection.weight @ encoder.projection.weight.transpose(0, 1)
    torch.testing.assert_close(row_gram, identity, atol=1e-5, rtol=1e-5)


def test_identity_projection_is_deterministic_for_its_config_seed() -> None:
    initial_weight = torch.zeros((5, 7), dtype=torch.float32)
    first = GeneIdentityEncoder(_small_identity_config(seed=9), initial_weight)
    torch.rand(11)  # Unrelated global RNG use must not affect initialization.
    second = GeneIdentityEncoder(_small_identity_config(seed=9), initial_weight)
    third = GeneIdentityEncoder(_small_identity_config(seed=10), initial_weight)

    assert torch.equal(first.projection.weight, second.projection.weight)
    assert not torch.equal(first.projection.weight, third.projection.weight)


def test_identity_encoder_rejects_wrong_asset_dtype_and_shape() -> None:
    config = _small_identity_config()
    with pytest.raises(TypeError, match="torch.float32"):
        GeneIdentityEncoder(config, torch.zeros((5, 7), dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        GeneIdentityEncoder(config, torch.zeros((4, 7), dtype=torch.float32))


def test_expression_encoder_is_pointwise_and_preserves_all_axes(monkeypatch) -> None:
    monkeypatch.setattr(expression_module, "NUM_GENES", 4)
    config = GeneExpressionEncoderConfig(hidden_dim=3, d_model=6)
    encoder = GeneExpressionEncoder(config)
    values = torch.tensor([[[-1.0], [0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0], [6.0]]])

    output = encoder(values)

    assert output.shape == (2, 4, 6)
    # Identical scalar values must receive identical encodings regardless of
    # their gene or batch position because both Linear layers are shared.
    duplicated = torch.full((1, 4, 1), 0.5)
    duplicate_output = encoder(duplicated)
    torch.testing.assert_close(
        duplicate_output[:, :1].expand_as(duplicate_output),
        duplicate_output,
    )


def test_expression_encoder_rejects_nonfloating_and_squeezed_inputs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(expression_module, "NUM_GENES", 4)
    encoder = GeneExpressionEncoder(GeneExpressionEncoderConfig(d_model=6))
    with pytest.raises(TypeError, match="floating dtype"):
        encoder(torch.zeros((1, 4, 1), dtype=torch.int64))
    with pytest.raises(ValueError, match="rank 3"):
        encoder(torch.zeros((4, 1), dtype=torch.float32))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def test_decoder_returns_typed_hurdle_parameters_without_squeezing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))
    with torch.no_grad():
        decoder.projection.weight.zero_()
        decoder.projection.bias.copy_(
            torch.tensor(
                [0.0, 0.0, _inverse_softplus(1.0 - decoder.config.min_scale)]
            )
        )

    result = decoder(torch.zeros((1, 4, 6), dtype=torch.float32))
    parameters = result.distribution_parameters

    assert result.point_prediction.shape == (1, 4, 1)
    assert result.point_prediction.dtype == torch.float32
    assert parameters.detection_logits.shape == (1, 4, 1)
    assert parameters.positive_location.shape == (1, 4, 1)
    assert parameters.positive_scale.shape == (1, 4, 1)
    assert parameters.detection_logits.dtype == torch.float32
    assert parameters.positive_location.dtype == torch.float32
    assert parameters.positive_scale.dtype == torch.float32
    torch.testing.assert_close(
        parameters.positive_scale,
        torch.ones_like(parameters.positive_scale),
    )

    # With detection probability 1/2 and a standard Normal positive component,
    # E[X] = 1/2 * E[N(0,1) | N(0,1)>0] = 1/sqrt(2*pi).
    expected_mean = torch.full_like(
        result.point_prediction,
        1.0 / math.sqrt(2.0 * math.pi),
    )
    torch.testing.assert_close(result.point_prediction, expected_mean)


def test_decoder_scale_floor_and_far_negative_mean_are_finite(monkeypatch) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6, min_scale=1e-3))
    with torch.no_grad():
        decoder.projection.weight.zero_()
        decoder.projection.bias.copy_(torch.tensor([0.0, -100.0, -100.0]))

    result = decoder(torch.zeros((2, 4, 6), dtype=torch.float32))

    assert torch.isfinite(result.point_prediction).all()
    assert (result.point_prediction >= 0.0).all()
    assert torch.isfinite(result.distribution_parameters.positive_scale).all()
    assert (
        result.distribution_parameters.positive_scale >= decoder.config.min_scale
    ).all()


def test_decoder_can_skip_point_prediction_without_evaluating_mean(
    monkeypatch,
) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))

    def unexpected_mean(*_args, **_kwargs):
        raise AssertionError("likelihood-only decoding must not compute the mean")

    monkeypatch.setattr(
        decoder_module,
        "_zero_truncated_normal_mean",
        unexpected_mean,
    )
    result = decoder(
        torch.zeros((2, 4, 6), dtype=torch.float32),
        compute_point_prediction=False,
    )

    assert result.point_prediction is None
    parameters = result.distribution_parameters
    assert parameters.detection_logits.shape == (2, 4, 1)
    assert parameters.positive_location.shape == (2, 4, 1)
    assert parameters.positive_scale.shape == (2, 4, 1)


def test_decoder_rejects_non_boolean_point_prediction_flag(monkeypatch) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))

    with pytest.raises(TypeError, match="compute_point_prediction"):
        decoder(
            torch.zeros((1, 4, 6), dtype=torch.float32),
            compute_point_prediction=1,
        )


def test_truncated_normal_mean_is_continuous_with_finite_gradients_at_cutoff() -> None:
    location = torch.tensor(
        [-10.001, -10.0, -9.999],
        dtype=torch.float32,
        requires_grad=True,
    )
    scale = torch.ones(3, dtype=torch.float32, requires_grad=True)

    positive_mean = decoder_module._zero_truncated_normal_mean(location, scale)

    assert torch.isfinite(positive_mean).all()
    assert torch.all(positive_mean > 0.0)
    # The two implementations differ only by the intentionally truncated tail
    # expansion; the transition must remain tiny relative to the ~0.098 mean.
    assert torch.max(torch.diff(positive_mean).abs()).item() < 5e-5

    positive_mean.sum().backward()
    assert location.grad is not None
    assert scale.grad is not None
    assert torch.isfinite(location.grad).all()
    assert torch.isfinite(scale.grad).all()
    assert torch.all(location.grad != 0.0)
    assert torch.all(scale.grad != 0.0)


def test_decoder_probability_head_remains_fp32_inside_cpu_autocast(
    monkeypatch,
) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    torch.manual_seed(211)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))
    hidden_states = torch.randn((2, 4, 6), dtype=torch.float32)
    reference = decoder(hidden_states)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        under_autocast = decoder(hidden_states)

    reference_parameters = reference.distribution_parameters
    autocast_parameters = under_autocast.distribution_parameters
    for reference_tensor, autocast_tensor in (
        (reference.point_prediction, under_autocast.point_prediction),
        (
            reference_parameters.detection_logits,
            autocast_parameters.detection_logits,
        ),
        (
            reference_parameters.positive_location,
            autocast_parameters.positive_location,
        ),
        (
            reference_parameters.positive_scale,
            autocast_parameters.positive_scale,
        ),
    ):
        assert autocast_tensor.dtype == torch.float32
        torch.testing.assert_close(
            autocast_tensor,
            reference_tensor,
            rtol=0.0,
            atol=0.0,
        )


def test_decoder_all_three_projection_channels_receive_gradients(monkeypatch) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))
    with torch.no_grad():
        decoder.projection.weight.zero_()
        decoder.projection.bias.zero_()
    hidden_states = torch.ones((2, 4, 6), dtype=torch.float32, requires_grad=True)

    result = decoder(hidden_states)
    result.point_prediction.sum().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert decoder.projection.weight.grad is not None
    channel_gradient_norms = decoder.projection.weight.grad.abs().sum(dim=1)
    assert torch.all(channel_gradient_norms > 0.0)


def test_real_decoder_and_hurdle_nll_train_all_three_channels(monkeypatch) -> None:
    num_genes = 4
    monkeypatch.setattr(decoder_module, "NUM_GENES", num_genes)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))
    criterion = TimeWeightedHurdleNLLLoss(LossConfig())
    with torch.no_grad():
        decoder.projection.weight.fill_(0.05)
        decoder.projection.bias.zero_()

    hidden_states = torch.ones(
        (1, num_genes, 6),
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[0.0], [1.0], [0.0], [2.0]]])
    diffusion_time = torch.tensor([0.5], dtype=torch.float32)
    diffusion_mask = torch.ones((1, num_genes), dtype=torch.bool)

    decoder_output = decoder(hidden_states, compute_point_prediction=False)
    assert decoder_output.point_prediction is None
    loss_output = criterion(
        decoder_output.distribution_parameters,
        target,
        diffusion_time,
        diffusion_mask,
    )
    loss_output.loss.backward()

    assert torch.isfinite(loss_output.loss)
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.count_nonzero(hidden_states.grad) > 0
    assert decoder.projection.weight.grad is not None
    assert decoder.projection.bias.grad is not None
    weight_gradient_by_channel = decoder.projection.weight.grad.abs().sum(dim=1)
    bias_gradient_by_channel = decoder.projection.bias.grad.abs()
    assert torch.isfinite(weight_gradient_by_channel).all()
    assert torch.isfinite(bias_gradient_by_channel).all()
    assert torch.all(weight_gradient_by_channel > 0.0)
    assert torch.all(bias_gradient_by_channel > 0.0)


def test_decoder_parameter_count_is_shared_three_channel_linear() -> None:
    d_model = 6
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=d_model))

    assert sum(parameter.numel() for parameter in decoder.parameters()) == (
        3 * d_model + 3
    )


def test_decoder_rejects_wrong_gene_or_feature_axis(monkeypatch) -> None:
    monkeypatch.setattr(decoder_module, "NUM_GENES", 4)
    decoder = GeneExpressionDecoder(DecoderConfig(d_model=6))
    with pytest.raises(ValueError, match="shape"):
        decoder(torch.zeros((1, 3, 6), dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        decoder(torch.zeros((1, 4, 5), dtype=torch.float32))
    with pytest.raises(TypeError, match="floating dtype"):
        decoder(torch.zeros((1, 4, 6), dtype=torch.int64))
