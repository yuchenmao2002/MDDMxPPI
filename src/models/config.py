"""
Typed Configuration and Cross-component Invariants
This module is the single source of truth for dimensions that cross component boundaries.
The processed gene vocabulary always contains 19,295 genes,
the Geneformer initialization has width 1,152,
and the model width is 512.
Component implementations must reject incompatible values instead of silently reshaping or truncating tensors.

It is also the source of truth for the *vocabulary* of the architecture
identifier: which backbone variants exist and how the identifier string is
spelled.  Deriving that identifier from a built model, and checking it against
what a configuration claims, lives in :mod:`src.models.architecture`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Tuple


NUM_GENES = 19_295
GENEFORMER_EMBEDDING_DIM = 1_152
DEFAULT_D_MODEL = 512
DEFAULT_EXPRESSION_HIDDEN_DIM = 32
DEFAULT_HEAD_DIM = 64
DEFAULT_RANDOM_FEATURES = 256

# PPI backbone assets: 19,295 genes on a rank-64 sphere, routed to four experts
# with a fixed top-2 support.  These match the shipped artifacts and are
# re-validated against both sidecar records at load time.
DEFAULT_PPI_EMBEDDING_RANK = 64
DEFAULT_PPI_NUM_EXPERTS = 4
DEFAULT_PPI_ROUTE_TOP_K = 2
DEFAULT_PPI_EXPERT_FFN_MULTIPLIER = 2
# Hidden width of the per-layer gate that turns the Fourier representation of
# the realized mask rate into one PPI prior share per attention head.
DEFAULT_PPI_GATE_HIDDEN_DIM = 64

# ---------------------------------------------------------------------------
# Architecture identifier
#
# The backbone segment comes first because the backbone is the axis experiments
# actually vary; the family and head segments are constants of this code
# revision.  The stack is homogeneous — L layers of one variant — so the segment
# is a single ``<variant>*<L>``.  Nothing anywhere compares against a frozen
# literal: identifiers are parsed, and their backbone segment is checked against
# the blocks that were really constructed.
#
#   ppil_full*6|masked-expression-diffusion-v2|hurdle-truncated-normal+inverse-t-nll
# ---------------------------------------------------------------------------
ARCHITECTURE_FAMILY = "masked-expression-diffusion-v2"
HEAD_SIGNATURE = "hurdle-truncated-normal+inverse-t-nll"

BACKBONE_VARIANT_PERFORMER = "performer"
BACKBONE_VARIANT_PPIL_ATTENTION = "ppil_attention"
BACKBONE_VARIANT_PPIL_FFN = "ppil_ffn"
BACKBONE_VARIANT_PPIL_FULL = "ppil_full"

BACKBONE_VARIANTS: Tuple[str, ...] = (
    BACKBONE_VARIANT_PERFORMER,
    BACKBONE_VARIANT_PPIL_ATTENTION,
    BACKBONE_VARIANT_PPIL_FFN,
    BACKBONE_VARIANT_PPIL_FULL,
)

# Variants whose blocks read the shared PPI assets.
PPI_BACKBONE_VARIANTS: Tuple[str, ...] = (
    BACKBONE_VARIANT_PPIL_ATTENTION,
    BACKBONE_VARIANT_PPIL_FFN,
    BACKBONE_VARIANT_PPIL_FULL,
)

_SIGNATURE_PATTERN = re.compile(r"^([A-Za-z0-9_]+)\*([1-9][0-9]*)$")


def backbone_signature(variant: str, num_layers: int) -> str:
    """Render the backbone segment for a homogeneous stack.

    The stack is always L layers of one variant, so this is a single
    ``<variant>*<L>``.  Mixing block types in one stack was a possibility the
    earlier design left open; it has since been dropped, and the identifier
    grammar was narrowed to match so that what an identifier can express and
    what the builder can build are the same set.
    """

    if variant not in BACKBONE_VARIANTS:
        raise ValueError(
            f"Unknown backbone variant {variant!r}; expected one of {BACKBONE_VARIANTS}."
        )
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}.")
    return f"{variant}*{num_layers}"


def parse_backbone_signature(text: str) -> Tuple[str, int]:
    """Inverse of :func:`backbone_signature`, rejecting unknown variants."""

    if not isinstance(text, str) or not text:
        raise ValueError("A backbone signature must be a non-empty string.")
    if "+" in text:
        raise ValueError(
            f"Backbone signature {text!r} names more than one block run. Stacks of "
            "mixed block types are no longer supported; a backbone is L layers of "
            "one variant."
        )
    match = _SIGNATURE_PATTERN.match(text)
    if match is None:
        raise ValueError(
            f"Malformed backbone signature {text!r}; expected '<variant>*<count>'."
        )
    variant, num_layers = match.group(1), int(match.group(2))
    if variant not in BACKBONE_VARIANTS:
        raise ValueError(
            f"Unknown backbone variant {variant!r} in signature {text!r}; "
            f"expected one of {BACKBONE_VARIANTS}."
        )
    return variant, num_layers


def architecture_id(signature: str) -> str:
    """Compose the full identifier from an already-validated backbone signature."""

    parse_backbone_signature(signature)
    return f"{signature}|{ARCHITECTURE_FAMILY}|{HEAD_SIGNATURE}"


def parse_architecture_id(text: str) -> Tuple[str, int]:
    """Validate an identifier against this code revision and return its backbone.

    This replaces the former equality test against one frozen literal.  The
    family and head segments must match the constants above, while the backbone
    segment is only required to name known variants — the caller then compares
    it against the blocks that were actually built.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("architecture_version must be a non-empty string.")
    segments = text.split("|")
    if len(segments) != 3:
        raise ValueError(
            "architecture_version must have three '|'-separated segments "
            f"(backbone|family|head); got {text!r}."
        )
    signature, family, head = segments
    if family != ARCHITECTURE_FAMILY:
        raise ValueError(
            f"Unsupported architecture family {family!r}; this code builds "
            f"{ARCHITECTURE_FAMILY!r}."
        )
    if head != HEAD_SIGNATURE:
        raise ValueError(
            f"Unsupported decoder/loss signature {head!r}; this code builds "
            f"{HEAD_SIGNATURE!r}."
        )
    return parse_backbone_signature(signature)


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

    This is also the common base for the PPIL backbone variants: every field
    below describes a dimension or discipline that all block types share.
    ``variant`` is a ``ClassVar``, so it does not become a dataclass field and
    never appears in ``asdict()`` — the serialized backbone configuration keeps
    exactly the shape it has today, and ``blocks/performer.py`` is unaffected.
    The variant a configuration object *is* therefore cannot drift from the
    variant it claims to be.
    """

    variant: ClassVar[str] = BACKBONE_VARIANT_PERFORMER

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


def _validate_ppi_attention_fields(ppi_rank: int, gate_hidden_dim: int) -> None:
    """Validate the knobs the PPI linear attention reads.

    There is deliberately no scalar strength for the PPI term: the share
    between the content and PPI halves of the augmented vector is the learned,
    per-head, per-cell gate ``lambda(p_t)``, not a configuration constant.
    """

    for name, value in (("ppi_rank", ppi_rank), ("gate_hidden_dim", gate_hidden_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def _validate_ppi_moe_fields(
    num_experts: int,
    route_top_k: int,
    expert_ffn_multiplier: int,
) -> None:
    for name, value in (
        ("num_experts", num_experts),
        ("route_top_k", route_top_k),
        ("expert_ffn_multiplier", expert_ffn_multiplier),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if route_top_k > num_experts:
        raise ValueError("route_top_k cannot exceed num_experts.")


@dataclass(frozen=True)
class PPILAttentionOnlyConfig(PerformerConfig):
    """PPI linear attention with the standard dense feed-forward network.

    Carries only the knobs this variant actually reads, so its serialized
    configuration cannot record a meaningless mixture-of-experts value.
    """

    variant: ClassVar[str] = BACKBONE_VARIANT_PPIL_ATTENTION

    ppi_rank: int = DEFAULT_PPI_EMBEDDING_RANK
    gate_hidden_dim: int = DEFAULT_PPI_GATE_HIDDEN_DIM

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_ppi_attention_fields(self.ppi_rank, self.gate_hidden_dim)

    @property
    def augmented_head_dim(self) -> int:
        """Width the random projection acts on: content half plus PPI half."""

        return self.head_dim + self.ppi_rank


@dataclass(frozen=True)
class PPILFeedForwardOnlyConfig(PerformerConfig):
    """Standard FAVOR+ attention with the statically routed mixture of experts."""

    variant: ClassVar[str] = BACKBONE_VARIANT_PPIL_FFN

    num_experts: int = DEFAULT_PPI_NUM_EXPERTS
    route_top_k: int = DEFAULT_PPI_ROUTE_TOP_K
    expert_ffn_multiplier: int = DEFAULT_PPI_EXPERT_FFN_MULTIPLIER

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_ppi_moe_fields(
            self.num_experts,
            self.route_top_k,
            self.expert_ffn_multiplier,
        )

    @property
    def expert_ffn_dim(self) -> int:
        """Hidden width of one routed expert."""

        return self.d_model * self.expert_ffn_multiplier


@dataclass(frozen=True)
class PPILFullConfig(PerformerConfig):
    """PPI linear attention together with the statically routed mixture of experts.

    The five fields are restated rather than inherited through multiple bases:
    dataclass multiple inheritance makes field order depend on the MRO, and the
    duplication here is five lines that stay obvious at a glance.
    """

    variant: ClassVar[str] = BACKBONE_VARIANT_PPIL_FULL

    ppi_rank: int = DEFAULT_PPI_EMBEDDING_RANK
    gate_hidden_dim: int = DEFAULT_PPI_GATE_HIDDEN_DIM
    num_experts: int = DEFAULT_PPI_NUM_EXPERTS
    route_top_k: int = DEFAULT_PPI_ROUTE_TOP_K
    expert_ffn_multiplier: int = DEFAULT_PPI_EXPERT_FFN_MULTIPLIER

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_ppi_attention_fields(self.ppi_rank, self.gate_hidden_dim)
        _validate_ppi_moe_fields(
            self.num_experts,
            self.route_top_k,
            self.expert_ffn_multiplier,
        )

    @property
    def augmented_head_dim(self) -> int:
        """Width the random projection acts on: content half plus PPI half."""

        return self.head_dim + self.ppi_rank

    @property
    def expert_ffn_dim(self) -> int:
        """Hidden width of one routed expert."""

        return self.d_model * self.expert_ffn_multiplier


BACKBONE_CONFIG_TYPES = {
    BACKBONE_VARIANT_PERFORMER: PerformerConfig,
    BACKBONE_VARIANT_PPIL_ATTENTION: PPILAttentionOnlyConfig,
    BACKBONE_VARIANT_PPIL_FFN: PPILFeedForwardOnlyConfig,
    BACKBONE_VARIANT_PPIL_FULL: PPILFullConfig,
}


@dataclass(frozen=True)
class PPIAssetConfig:
    """Location and contract of the two shared PPI backbone assets.

    Both artifacts are indexed by gene over the same 19,295-row axis the
    denoiser uses, so no identifier remapping exists anywhere in the model.  The
    tensors are carried inside the training checkpoint, so these paths are read
    once at training-time construction and never at inference.
    """

    embedding_path: Path = Path(
        "data/processed/PPI/spherical_ppi_tau700_k4_b1_r64.safetensors"
    )
    embedding_manifest_path: Path = Path(
        "data/processed/PPI/spherical_ppi_tau700_k4_b1_r64.json"
    )
    routing_path: Path = Path("data/processed/PPI/ppi_moe_routing.safetensors")
    routing_manifest_path: Path = Path("data/processed/PPI/ppi_moe_routing.json")
    num_genes: int = NUM_GENES
    embedding_rank: int = DEFAULT_PPI_EMBEDDING_RANK
    num_experts: int = DEFAULT_PPI_NUM_EXPERTS
    route_top_k: int = DEFAULT_PPI_ROUTE_TOP_K
    verify_sha256: bool = True

    def __post_init__(self) -> None:
        if self.num_genes != NUM_GENES:
            raise ValueError(
                f"v2 requires num_genes={NUM_GENES}, got {self.num_genes}."
            )
        for name in (
            "embedding_path",
            "embedding_manifest_path",
            "routing_path",
            "routing_manifest_path",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty path.")
        _validate_ppi_moe_fields(self.num_experts, self.route_top_k, 1)
        if (
            isinstance(self.embedding_rank, bool)
            or not isinstance(self.embedding_rank, int)
            or self.embedding_rank <= 0
        ):
            raise ValueError("embedding_rank must be a positive integer.")


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
    ``initial_detection_probability`` is the empirical prior the detection head
    starts from: the readout is initialized so that ``sigmoid(eta)`` equals it
    exactly for every gene before training.  It tracks the processed dataset's
    strictly-positive rate, measured at 0.0955 over 3,000 randomly sampled cells
    of the 590,929-cell PBS matrix, and rounded to 0.1.
    These values are fixed architecture semantics, not experiment knobs.
    """

    d_model: int = DEFAULT_D_MODEL
    output_dim: int = 3
    kind: str = "hurdle_truncated_normal"
    positive_distribution: str = "zero_truncated_normal"
    min_scale: float = 1e-3
    initial_detection_probability: float = 0.1

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
        if (
            isinstance(self.initial_detection_probability, bool)
            or not isinstance(self.initial_detection_probability, (int, float))
            or not math.isfinite(self.initial_detection_probability)
            or not 0.0 < self.initial_detection_probability < 1.0
        ):
            raise ValueError(
                "initial_detection_probability must be a finite number in (0,1)."
            )


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
    """Complete configuration with cross-component validation.

    The backbone comes first because it is the axis experiments vary.
    ``backbone_variant`` has no default: every construction states which of the
    four variants it builds, and ``__post_init__`` checks that claim against the
    concrete type of ``backbone`` so the two can never disagree.

    ``architecture_version`` is a derived property rather than a field.  It used
    to be a global constant disguised as configuration — a field whose only
    permitted value was one literal — which meant it could not describe the
    backbone at all.  Deriving it removes the possibility of a stored identifier
    that disagrees with the configuration it sits next to.
    """

    backbone: PerformerConfig
    backbone_variant: str
    ppi: Optional[PPIAssetConfig] = None
    gene_identity: GeneIdentityEncoderConfig = field(
        default_factory=GeneIdentityEncoderConfig
    )
    gene_expression: GeneExpressionEncoderConfig = field(
        default_factory=GeneExpressionEncoderConfig
    )
    forward_process: ForwardProcessConfig = field(default_factory=ForwardProcessConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        if self.backbone_variant not in BACKBONE_VARIANTS:
            raise ValueError(
                f"Unknown backbone_variant={self.backbone_variant!r}; expected one "
                f"of {BACKBONE_VARIANTS}."
            )
        declared_type = type(self.backbone)
        if declared_type.variant != self.backbone_variant:
            raise ValueError(
                f"backbone_variant={self.backbone_variant!r} does not match the "
                f"supplied {declared_type.__name__}, which builds "
                f"{declared_type.variant!r}."
            )

        widths = {
            self.gene_identity.d_model,
            self.gene_expression.d_model,
            self.backbone.d_model,
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
        if self.backbone.feature_redraw_interval is not None:
            raise ValueError(
                "v2 uses fixed Performer projections; redraw scheduling is not "
                "available until a synchronized trainer implementation exists."
            )

        needs_ppi = self.backbone_variant in PPI_BACKBONE_VARIANTS
        if needs_ppi and self.ppi is None:
            raise ValueError(
                f"backbone_variant={self.backbone_variant!r} reads the shared PPI "
                "assets, so a PPIAssetConfig is required."
            )
        if not needs_ppi and self.ppi is not None:
            raise ValueError(
                f"backbone_variant={self.backbone_variant!r} does not read the PPI "
                "assets; leave ppi unset so the configuration cannot imply an "
                "asset dependency the model does not have."
            )
        if self.ppi is not None:
            if self.ppi.num_genes != self.forward_process.num_genes:
                raise ValueError("PPI asset and forward-process gene counts must match.")
            declared_rank = getattr(self.backbone, "ppi_rank", None)
            if declared_rank is not None and declared_rank != self.ppi.embedding_rank:
                raise ValueError(
                    f"backbone ppi_rank={declared_rank} does not match the asset "
                    f"embedding_rank={self.ppi.embedding_rank}."
                )
            for name in ("num_experts", "route_top_k"):
                declared = getattr(self.backbone, name, None)
                if declared is not None and declared != getattr(self.ppi, name):
                    raise ValueError(
                        f"backbone {name}={declared} does not match the routing "
                        f"asset {name}={getattr(self.ppi, name)}."
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

    @property
    def backbone_signature(self) -> str:
        """Backbone segment this configuration claims the stack will have."""

        return backbone_signature(self.backbone_variant, self.backbone.num_layers)

    @property
    def architecture_version(self) -> str:
        """Derived identifier, backbone segment first."""

        return architecture_id(self.backbone_signature)
