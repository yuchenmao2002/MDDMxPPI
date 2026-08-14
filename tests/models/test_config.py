"""Configuration-only tests; these do not require model construction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.models.config import (
    ARCHITECTURE_VERSION,
    DEFAULT_D_MODEL,
    GENEFORMER_EMBEDDING_DIM,
    NUM_GENES,
    DecoderConfig,
    GeneIdentityEncoderConfig,
    LossConfig,
    MaskedDiffusionModelConfig,
    PerformerConfig,
)


def test_confirmed_v2_dimensions_and_probabilistic_head_are_consistent() -> None:
    config = MaskedDiffusionModelConfig(performer=PerformerConfig(num_layers=2))

    assert config.gene_identity.num_genes == NUM_GENES == 19_295
    assert config.gene_identity.source_dim == GENEFORMER_EMBEDDING_DIM == 1_152
    assert config.gene_identity.d_model == DEFAULT_D_MODEL == 512
    assert config.performer.num_heads == 8
    assert config.performer.ffn_dim == 2_048
    assert config.performer.sequence_chunk_size == 8_192
    assert config.performer.activation_checkpointing is False
    assert config.decoder.kind == "hurdle_truncated_normal"
    assert config.decoder.output_dim == 3
    assert config.decoder.positive_distribution == "zero_truncated_normal"
    assert config.decoder.min_scale == pytest.approx(1e-3)
    assert config.loss.kind == "time_weighted_hurdle_nll"
    assert config.loss.reduction == "cell_gene_mean"
    assert config.loss.time_weighting == "inverse_t"
    assert config.architecture_version == ARCHITECTURE_VERSION
    assert "v2-hurdle-truncated-normal" in ARCHITECTURE_VERSION


def test_v2_rejects_cross_component_width_drift() -> None:
    config = MaskedDiffusionModelConfig(performer=PerformerConfig(num_layers=1))
    incompatible_identity = replace(config.gene_identity, d_model=256)

    with pytest.raises(ValueError, match="d_model=512"):
        replace(config, gene_identity=incompatible_identity)


def test_v2_requires_trainable_bias_free_gene_projection() -> None:
    base_identity = GeneIdentityEncoderConfig()
    performer = PerformerConfig(num_layers=1)

    with pytest.raises(ValueError, match="must remain trainable"):
        MaskedDiffusionModelConfig(
            performer=performer,
            gene_identity=replace(base_identity, trainable=False),
        )
    with pytest.raises(ValueError, match="bias-free"):
        MaskedDiffusionModelConfig(
            performer=performer,
            gene_identity=replace(base_identity, projection_bias=True),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"output_dim": 1}, "three channels"),
        ({"kind": "linear"}, "hurdle_truncated_normal"),
        ({"positive_distribution": "normal"}, "zero_truncated_normal"),
        ({"min_scale": 0.0}, "positive finite"),
        ({"min_scale": -1.0}, "positive finite"),
        ({"min_scale": float("inf")}, "positive finite"),
        ({"min_scale": float("nan")}, "positive finite"),
    ],
)
def test_decoder_config_rejects_noncanonical_or_unsafe_values(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DecoderConfig(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "plain_masked_mse"}, "time_weighted_hurdle_nll"),
        ({"reduction": "global_masked_token_mean"}, "cell_gene_mean"),
        ({"time_weighting": "none"}, "inverse_t"),
    ],
)
def test_loss_config_rejects_legacy_objective_semantics(
    changes: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LossConfig(**changes)
