"""Architecture identity: what a model *is*, derived from what was built.

``src/models/config.py`` owns the vocabulary — which backbone variants exist and
how the identifier string is spelled.  This module owns the other half: reading
the identity off a constructed backbone and checking it against what a
configuration claimed.

The identifier is three ``|``-separated segments with the backbone first::

    ppil_full*6|masked-expression-diffusion-v2|hurdle-truncated-normal+inverse-t-nll

The backbone leads because it is the axis experiments vary; the remaining two
segments are constants of this code revision.  The backbone segment names the
*actual* block type and how many of them were built.  A stack always holds one
variant: mixing block types was a possibility an earlier design left open, and
it has since been dropped, so a mixed stack is now an error rather than
something the identifier describes.

Three consumers need different strictness, and separating them is the point of
this module:

1. **Resume** needs exact equality.  It keeps comparing the whole serialized
   ``model_config`` plus the training signature; nothing here loosens that.
2. **Inference loading** needs structural compatibility — can this code hold
   these weights.  That is :func:`verify_backbone_matches` against the
   identifier recovered from the checkpoint, backed by the strict state-dict
   load.
3. **Experiment tracking** needs readability, which the string provides.

The identifier is never compared against a single frozen literal.  It is
parsed, and its backbone segment is checked against the blocks that were really
constructed.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Type

from torch import nn

from src.models.blocks.base import DenoiserBlock
from src.models.blocks.performer import PerformerBlock
from src.models.blocks.ppil_attention_only import PPILAttentionOnlyBlock
from src.models.blocks.ppil_ffn_only import PPILFeedForwardOnlyBlock
from src.models.blocks.ppil_full import PPILFullBlock
from src.models.config import (
    ARCHITECTURE_FAMILY,
    BACKBONE_VARIANT_PERFORMER,
    BACKBONE_VARIANT_PPIL_ATTENTION,
    BACKBONE_VARIANT_PPIL_FFN,
    BACKBONE_VARIANT_PPIL_FULL,
    BACKBONE_VARIANTS,
    HEAD_SIGNATURE,
    MaskedDiffusionModelConfig,
    architecture_id,
    backbone_signature,
    parse_architecture_id,
    parse_backbone_signature,
)


# The tag registry lives here rather than on the block classes because
# ``blocks/performer.py`` is deliberately not modified.  Mapping the concrete
# type keeps the tag out of the blocks entirely, so a block cannot mislabel
# itself.
_BLOCK_TAGS: Dict[Type[DenoiserBlock], str] = {
    PerformerBlock: BACKBONE_VARIANT_PERFORMER,
    PPILAttentionOnlyBlock: BACKBONE_VARIANT_PPIL_ATTENTION,
    PPILFeedForwardOnlyBlock: BACKBONE_VARIANT_PPIL_FFN,
    PPILFullBlock: BACKBONE_VARIANT_PPIL_FULL,
}

BLOCK_TYPES: Dict[str, Type[DenoiserBlock]] = {
    tag: block_type for block_type, tag in _BLOCK_TAGS.items()
}


def block_tag(block: nn.Module) -> str:
    """Return the registered variant tag for one constructed block.

    Exact type match, not ``isinstance``: a subclass is a different
    architecture and must register its own tag rather than inherit an
    identifier that would misdescribe its weights.
    """

    tag = _BLOCK_TAGS.get(type(block))
    if tag is None:
        raise KeyError(
            f"Block type {type(block).__name__} has no registered architecture "
            f"tag; register it in {__name__}._BLOCK_TAGS so checkpoints built "
            "from it can be identified."
        )
    return tag


def backbone_signature_from_blocks(blocks: Sequence[nn.Module]) -> str:
    """Derive the backbone signature from the blocks that were actually built.

    The stack must be homogeneous.  Rejecting a mixed one here is what keeps the
    set of architectures an identifier can express equal to the set the builder
    can produce, so a stack assembled by hand cannot slip through carrying a
    signature that names only its first block type.
    """

    if len(blocks) == 0:
        raise ValueError("A backbone must contain at least one block.")
    tags = [block_tag(block) for block in blocks]
    distinct = sorted(set(tags))
    if len(distinct) != 1:
        raise ValueError(
            f"A backbone must be L layers of one variant, but this stack mixes "
            f"{distinct}. Stacks of mixed block types are no longer supported."
        )
    return backbone_signature(tags[0], len(tags))


def backbone_signature_from_model(backbone: nn.Module) -> str:
    """Derive the backbone signature from a live :class:`DenoiserBackbone`."""

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise TypeError(
            "backbone must expose an ordered 'blocks' module list to be identified."
        )
    return backbone_signature_from_blocks(list(blocks))


def architecture_version_from_model(backbone: nn.Module) -> str:
    """Full identifier for a constructed backbone."""

    return architecture_id(backbone_signature_from_model(backbone))


def verify_backbone_matches(
    backbone: nn.Module,
    *,
    expected_signature: str,
    context: str,
) -> None:
    """Raise unless the constructed stack has the expected backbone signature.

    This is the check that makes the identifier meaningful: a configuration's
    claim about its own architecture is worth nothing until it has been
    compared against the blocks that were really instantiated.
    """

    parse_backbone_signature(expected_signature)
    actual = backbone_signature_from_model(backbone)
    if actual != expected_signature:
        raise ValueError(
            f"{context}: the constructed backbone is {actual!r} but "
            f"{expected_signature!r} was expected."
        )


def verify_architecture_version(
    version: str,
    *,
    config: MaskedDiffusionModelConfig,
    context: str,
) -> Tuple[str, int]:
    """Validate a stored identifier against this code and against ``config``.

    Returns the parsed ``(variant, num_layers)`` so a caller can report it.
    Parsing rejects an unknown family, an unknown decoder/loss signature and
    unknown backbone variants; the equality check then pins the identifier to
    the configuration recovered alongside it.
    """

    backbone = parse_architecture_id(version)
    expected = config.architecture_version
    if version != expected:
        raise ValueError(
            f"{context}: checkpoint architecture {version!r} does not match the "
            f"architecture its own model configuration describes, {expected!r}."
        )
    return backbone


__all__ = [
    "ARCHITECTURE_FAMILY",
    "BACKBONE_VARIANTS",
    "BLOCK_TYPES",
    "HEAD_SIGNATURE",
    "architecture_id",
    "architecture_version_from_model",
    "backbone_signature",
    "backbone_signature_from_blocks",
    "backbone_signature_from_model",
    "block_tag",
    "parse_architecture_id",
    "parse_backbone_signature",
    "verify_architecture_version",
    "verify_backbone_matches",
]
