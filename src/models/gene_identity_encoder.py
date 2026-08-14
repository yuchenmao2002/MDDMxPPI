"""Gene Identity Encoder for the fixed 19,295-gene PBS vocabulary.

The encoder owns the full trainable Geneformer-initialized source table and a
shared projection from 1,152 to the confirmed model width of 512.  It does not
accept a per-batch gene-index tensor because every model sample uses exactly the
same ordered axis ``0..19294``.  Its unbatched output must be broadcast with
``expand`` by the top-level model rather than copied with ``repeat``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from src.models.config import GeneIdentityEncoderConfig
from src.utils.checkpoint import (
    GeneEmbeddingAssetMetadata,
    load_geneformer_embedding_asset,
)


class GeneIdentityEncoder(nn.Module):
    """Map the fixed gene vocabulary to trainable 512-dimensional identities.

    Parameters
    ----------
    config:
        Dimension, asset and trainability contract.
    initial_weight:
        Float tensor of shape ``[19295,1152]`` loaded and validated from the
        completed Geneformer safetensors asset.  Implementations must copy this
        tensor into ``nn.Embedding`` and must not rerun the 59-gene donor
        initialization procedure.

    Notes
    -----
    The projection is ``Linear(1152,512,bias=False)``.  It should use a
    deterministic orthogonal initialization and remain trainable.  The source
    embedding is also trainable, as explicitly required by the model design.
    """

    def __init__(
        self,
        config: GeneIdentityEncoderConfig,
        initial_weight: Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.asset_metadata: Optional[GeneEmbeddingAssetMetadata] = None

        if config.projection_bias:
            raise ValueError("The gene-identity projection must use bias=False.")
        if isinstance(config.projection_seed, bool) or not isinstance(
            config.projection_seed, int
        ):
            raise TypeError("projection_seed must be an integer.")
        if not isinstance(initial_weight, Tensor):
            raise TypeError(
                "initial_weight must be a torch.Tensor, "
                f"got {type(initial_weight).__name__}."
            )
        expected_shape = (config.num_genes, config.source_dim)
        if tuple(initial_weight.shape) != expected_shape:
            raise ValueError(
                "initial_weight must have shape "
                f"{expected_shape}, got {tuple(initial_weight.shape)}."
            )
        if initial_weight.dtype != torch.float32:
            raise TypeError(
                "initial_weight must have dtype torch.float32, "
                f"got {initial_weight.dtype}."
            )
        if initial_weight.device.type != "cpu":
            raise ValueError(
                "initial_weight must be on CPU; construct the encoder first and "
                f"then move the module to {initial_weight.device}."
            )
        if initial_weight.layout != torch.strided:
            raise TypeError(
                "initial_weight must use torch.strided layout, "
                f"got {initial_weight.layout}."
            )
        if not bool(torch.isfinite(initial_weight).all().item()):
            raise ValueError("initial_weight must contain only finite values.")

        # from_pretrained with a supplied weight does not run an Embedding
        # initializer.  Clone so the model owns its parameter storage rather
        # than sharing mutable storage with the asset loader/caller.
        owned_weight = initial_weight.detach().contiguous().clone()
        self.embedding = nn.Embedding.from_pretrained(
            owned_weight,
            freeze=not config.trainable,
        )

        # nn.Linear normally consumes the global CPU RNG in reset_parameters.
        # fork_rng restores that state immediately; the actual orthogonal draw
        # uses its own seeded Generator, so construction is deterministic and
        # leaves the caller's global RNG stream untouched.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(
                config.source_dim,
                config.d_model,
                bias=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        projection_generator = torch.Generator(device="cpu")
        projection_generator.manual_seed(config.projection_seed)
        with torch.no_grad():
            nn.init.orthogonal_(
                self.projection.weight,
                gain=1.0,
                generator=projection_generator,
            )

        # The projection is always learnable.  Source-table trainability is the
        # explicit config choice (True in the confirmed configuration).
        self.projection.weight.requires_grad_(True)

    @classmethod
    def from_config(cls, config: GeneIdentityEncoderConfig) -> "GeneIdentityEncoder":
        """Load the audited asset named by ``config`` and construct the encoder."""

        project_root = Path(__file__).resolve().parents[2]
        initial_weight, metadata = load_geneformer_embedding_asset(
            config.weights_path,
            config.manifest_path,
            tensor_key=config.tensor_key,
            expected_shape=(config.num_genes, config.source_dim),
            verify_sha256=config.verify_sha256,
            project_root=project_root,
        )
        encoder = cls(config, initial_weight)
        encoder.asset_metadata = metadata
        return encoder

    def forward(self) -> Tensor:
        """Return gene identities with shape ``[19295,512]``.

        The method has no batch-dependent input.  It must preserve the exact
        mapping row order and must not cache a detached projected result because
        both the source table and projection are updated during training.
        """

        return self.projection(self.embedding.weight)
