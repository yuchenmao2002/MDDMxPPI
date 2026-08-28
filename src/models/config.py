"""
Typed Configuration and Cross-component Invariants
This module is the single source of truth for dimensions that cross component boundaries.
The processed gene vocabulary always contains 19,295 genes,
the Geneformer initialization has width 1,152,
and the model width is 512.
Component implementations must reject incompatible values instead of silently reshaping or truncating tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


NUM_GENES = 19_295
GENEFORMER_EMBEDDING_DIM = 1_152
DEFAULT_D_MODEL = 512
DEFAULT_EXPRESSION_HIDDEN_DIM = 32
DEFAULT_HEAD_DIM = 64
DEFAULT_RANDOM_FEATURES = 256
ARCHITECTURE_VERSION = "masked-expression-diffusion-v2-hurdle-truncated-normal"


@dataclass(frozen=True)
class GeneIdentityEncoderConfig:
    """
    Configuration for the fixed-vocabulary Gene Identity Encoder.
    """

    weights_path: Path = Path(
        "data/processed/Geneformer/hgnc_V2_embeddings.safetensors"
    )
    manifest_path: Path = Path(
        "data/processed/Geneformer/hgnc_V2_embeddings_manifest.json"
    )
    tensor_key: str = "weight"
    num_genes: int = NUM_GENES
    source_dim: int = GENEFORMER_EMBEDDING_DIM
    d_model: int = DEFAULT_D_MODEL
    trainable: bool = False
    projection_bias: bool = False
    projection_seed: int = 0
    verify_sha256: bool = True

    def __post_init__(self) -> None:
        if self.num_genes != NUM_GENES:
            raise ValueError(
                f"v2 requires num_genes={NUM_GENES}, got {self.num_genes}."
            )
        if self.source_dim != GENEFORMER_EMBEDDING_DIM:
            raise ValueError(
                "Geneformer initialization width must be "
                f"{GENEFORMER_EMBEDDING_DIM}, got {self.source_dim}."
            )
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if not self.tensor_key:
            raise ValueError("tensor_key must be non-empty.")


@dataclass(frozen=True)
class GeneExpressionEncoderConfig:
    """Configuration for the shared pointwise expression-value MLP."""

    input_dim: int = 1
    hidden_dim: int = DEFAULT_EXPRESSION_HIDDEN_DIM
    d_model: int = DEFAULT_D_MODEL
    activation: str = "silu"
    bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim != 1:
            raise ValueError("The v2 expression encoder requires scalar inputs.")
        if self.hidden_dim <= 0 or self.d_model <= 0:
            raise ValueError("hidden_dim and d_model must be positive.")
        if self.activation != "silu":
            raise ValueError("The v2 expression encoder activation is fixed to SiLU.")


@dataclass(frozen=True)
class ForwardProcessConfig:
    """Configuration for continuous-time independent absorbing masking."""

    num_genes: int = NUM_GENES
    time_distribution: str = "uniform"
    force_at_least_one_mask: bool = False

    def __post_init__(self) -> None:
        if self.num_genes != NUM_GENES:
            raise ValueError(f"v2 requires num_genes={NUM_GENES}.")
        if self.time_distribution != "uniform":
            raise ValueError("v2 supports only per-cell Uniform[0,1) time sampling.")
        if self.force_at_least_one_mask:
            raise ValueError(
                "v2 must not force a masked position because that changes q(x_t|x_0)."
            )


@dataclass(frozen=True)
class PerformerConfig:
    """
    Configuration shared by all standard Performer blocks.
    Random-feature projections are fixed by default and stored as persistent buffers.
    """

    num_layers: int
    d_model: int = DEFAULT_D_MODEL
    head_dim: int = DEFAULT_HEAD_DIM
    num_random_features: int = DEFAULT_RANDOM_FEATURES
    ffn_multiplier: int = 4
    dropout: float = 0.0
    feature_epsilon: float = 1e-6
    sequence_chunk_size: int = 8_192
    projection_seed: int = 0
    feature_redraw_interval: Optional[int] = None
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.d_model <= 0 or self.head_dim <= 0:
            raise ValueError("d_model and head_dim must be positive.")
        if self.d_model % self.head_dim != 0:
            raise ValueError("d_model must be divisible by head_dim.")
        if self.num_random_features <= 0:
            raise ValueError("num_random_features must be positive.")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1).")
        if self.feature_epsilon <= 0.0:
            raise ValueError("feature_epsilon must be positive.")
        if self.sequence_chunk_size <= 0:
            raise ValueError("sequence_chunk_size must be positive.")
        if (
            self.feature_redraw_interval is not None
            and self.feature_redraw_interval <= 0
        ):
            raise ValueError("feature_redraw_interval must be positive or None.")

    @property
    def num_heads(self) -> int:
        """Number of attention heads; eight for the confirmed v2 dimensions."""

        return self.d_model // self.head_dim

    @property
    def ffn_dim(self) -> int:
        """Hidden width of the position-wise feed-forward network."""

        return self.d_model * self.ffn_multiplier


@dataclass(frozen=True)
class DecoderConfig:
    """
    Configuration for the shared probabilistic expression readout.
    The three raw channels are, in order,
    the positive/detection logit,
    the positive-component location,
    and an unconstrained scale parameter.
    The decoder transforms the last channel with ``min_scale + softplus(raw)``
    and models positive values with a Normal distribution truncated to (0,inf).
    These values are fixed architecture semantics, not experiment knobs.
    """

    d_model: int = DEFAULT_D_MODEL
    output_dim: int = 3
    kind: str = "hurdle_truncated_normal"
    positive_distribution: str = "zero_truncated_normal"
    min_scale: float = 1e-3

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.output_dim != 3:
            raise ValueError("The v2 hurdle decoder must output three channels.")
        if self.kind != "hurdle_truncated_normal":
            raise ValueError("The v2 decoder kind must be hurdle_truncated_normal.")
        if self.positive_distribution != "zero_truncated_normal":
            raise ValueError(
                "The v2 positive distribution must be zero_truncated_normal."
            )
        if (
            isinstance(self.min_scale, bool)
            or not isinstance(self.min_scale, (int, float))
            or not math.isfinite(self.min_scale)
            or self.min_scale <= 0.0
        ):
            raise ValueError("min_scale must be a positive finite number.")


@dataclass(frozen=True)
class LossConfig:
    """Configuration for the time-weighted hurdle negative log-likelihood."""

    kind: str = "time_weighted_hurdle_nll"
    reduction: str = "cell_gene_mean"
    time_weighting: str = "inverse_t"

    def __post_init__(self) -> None:
        if self.kind != "time_weighted_hurdle_nll":
            raise ValueError("v2 loss kind must be time_weighted_hurdle_nll.")
        if self.reduction != "cell_gene_mean":
            raise ValueError("v2 loss reduction must be cell_gene_mean.")
        if self.time_weighting != "inverse_t":
            raise ValueError("v2 loss time weighting must be inverse_t.")


@dataclass(frozen=True)
class MaskedDiffusionModelConfig:
    """Complete configuration with cross-component validation."""

    performer: PerformerConfig
    gene_identity: GeneIdentityEncoderConfig = field(
        default_factory=GeneIdentityEncoderConfig
    )
    gene_expression: GeneExpressionEncoderConfig = field(
        default_factory=GeneExpressionEncoderConfig
    )
    forward_process: ForwardProcessConfig = field(default_factory=ForwardProcessConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        widths = {
            self.gene_identity.d_model,
            self.gene_expression.d_model,
            self.performer.d_model,
            self.decoder.d_model,
        }
        if widths != {DEFAULT_D_MODEL}:
            raise ValueError(
                f"All v2 components must use d_model={DEFAULT_D_MODEL}; got {widths}."
            )
        if self.gene_identity.num_genes != self.forward_process.num_genes:
            raise ValueError("Gene vocabulary and forward-process sizes must match.")
        if self.gene_identity.projection_bias:
            raise ValueError("The v2 1152-to-512 gene projection must be bias-free.")
        if self.performer.feature_redraw_interval is not None:
            raise ValueError(
                "v2 uses fixed Performer projections; redraw scheduling is not "
                "available until a synchronized trainer implementation exists."
            )
        if (
            self.decoder.kind != "hurdle_truncated_normal"
            or self.decoder.positive_distribution != "zero_truncated_normal"
            or self.loss.kind != "time_weighted_hurdle_nll"
            or self.loss.reduction != "cell_gene_mean"
            or self.loss.time_weighting != "inverse_t"
        ):
            raise ValueError(
                "v2 requires the hurdle truncated-Normal decoder together with "
                "the inverse-time-weighted cell-gene-mean NLL."
            )
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported architecture_version={self.architecture_version!r}."
            )
