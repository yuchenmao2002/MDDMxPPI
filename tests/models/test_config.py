"""Configuration-only tests; these do not require model construction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.models.config import (
    ARCHITECTURE_FAMILY,
    DEFAULT_D_MODEL,
    GENEFORMER_EMBEDDING_DIM,
    HEAD_SIGNATURE,
    NUM_GENES,
    DecoderConfig,
    GeneIdentityEncoderConfig,
    LossConfig,
    MaskedDiffusionModelConfig,
    PPIAssetConfig,
    PPILAttentionOnlyConfig,
    PPILFeedForwardOnlyConfig,
    PPILFullConfig,
    PerformerConfig,
    parse_architecture_id,
)


def _performer(**overrides) -> MaskedDiffusionModelConfig:
    layers = overrides.pop("num_layers", 2)
    return MaskedDiffusionModelConfig(
        backbone=PerformerConfig(num_layers=layers),
        backbone_variant="performer",
        **overrides,
    )


def test_confirmed_v2_dimensions_and_probabilistic_head_are_consistent() -> None:
    config = _performer(num_layers=2)

    assert config.gene_identity.num_genes == NUM_GENES == 19_295
    assert config.gene_identity.source_dim == GENEFORMER_EMBEDDING_DIM == 1_152
    assert config.gene_identity.d_model == DEFAULT_D_MODEL == 512
    assert config.backbone.num_heads == 8
    assert config.backbone.ffn_dim == 2_048
    assert config.backbone.sequence_chunk_size == 8_192
    assert config.backbone.activation_checkpointing is True
    assert config.decoder.kind == "hurdle_truncated_normal"
    assert config.decoder.output_dim == 3
    assert config.decoder.positive_distribution == "zero_truncated_normal"
    assert config.decoder.min_scale == pytest.approx(1e-3)
    assert config.loss.kind == "time_weighted_hurdle_nll"
    assert config.loss.reduction == "cell_gene_mean"
    assert config.loss.time_weighting == "inverse_t"
    assert config.architecture_version == (
        f"performer*2|{ARCHITECTURE_FAMILY}|{HEAD_SIGNATURE}"
    )
    assert parse_architecture_id(config.architecture_version) == ("performer", 2)


def test_architecture_identifier_puts_the_backbone_first_for_every_variant() -> None:
    variants = {
        "performer": (PerformerConfig, None),
        "ppil_attention": (PPILAttentionOnlyConfig, PPIAssetConfig()),
        "ppil_ffn": (PPILFeedForwardOnlyConfig, PPIAssetConfig()),
        "ppil_full": (PPILFullConfig, PPIAssetConfig()),
    }
    seen = set()
    for variant, (backbone_type, ppi) in variants.items():
        config = MaskedDiffusionModelConfig(
            backbone=backbone_type(num_layers=6),
            backbone_variant=variant,
            ppi=ppi,
        )
        assert config.architecture_version.startswith(f"{variant}*6|")
        assert config.backbone_signature == f"{variant}*6"
        seen.add(config.architecture_version)
    # The four variants must be distinguishable by their identifier alone.
    assert len(seen) == 4


def test_declared_variant_must_match_the_backbone_configuration_type() -> None:
    with pytest.raises(ValueError, match="does not match the supplied"):
        MaskedDiffusionModelConfig(
            backbone=PerformerConfig(num_layers=1),
            backbone_variant="ppil_full",
            ppi=PPIAssetConfig(),
        )


def test_ppi_assets_are_required_by_exactly_the_variants_that_read_them() -> None:
    with pytest.raises(ValueError, match="requires|required"):
        MaskedDiffusionModelConfig(
            backbone=PPILFullConfig(num_layers=1),
            backbone_variant="ppil_full",
        )
    with pytest.raises(ValueError, match="does not read the PPI assets"):
        MaskedDiffusionModelConfig(
            backbone=PerformerConfig(num_layers=1),
            backbone_variant="performer",
            ppi=PPIAssetConfig(),
        )


def test_v2_rejects_cross_component_width_drift() -> None:
    config = _performer(num_layers=1)
    incompatible_identity = replace(config.gene_identity, d_model=256)

    with pytest.raises(ValueError, match="d_model=512"):
        replace(config, gene_identity=incompatible_identity)


def test_v2_requires_a_bias_free_gene_projection() -> None:
    base_identity = GeneIdentityEncoderConfig()

    with pytest.raises(ValueError, match="bias-free"):
        _performer(
            num_layers=1,
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
