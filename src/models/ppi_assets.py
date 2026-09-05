"""Shared PPI assets owned by the denoising backbone.

Both artifacts are indexed by gene over the same 19,295-row axis as the
denoiser's hidden states, and every block in one backbone reads the same two
tables.  They are therefore owned once, by the backbone, and handed to the
blocks by reference: the state dict contains exactly one copy no matter how
many layers are stacked.

The tensors are **persistent** buffers.  This follows the position already
taken for the 88.9 MB Geneformer identity table, which also travels inside the
checkpoint: the marginal cost here is 5.17 MiB, and in exchange a trained
checkpoint is self-contained.  Inference reconstructs the model with
:meth:`PPIAssets.empty` and lets the strict state-dict load supply the values,
exactly as ``_build_model_from_state`` already avoids reopening the Geneformer
asset.

Two derived quantities are kept in *non-persistent* buffers next to them: the
per-expert routing index, and the spherical embedding with its free rows zeroed.
The routing index cannot be persistent — each expert's index list has its own
length, and the placeholder built by :meth:`empty` would have the wrong lengths,
so a strict load would fail on shape rather than overwrite.  A load-state-dict
post hook rebuilds both once the real tables arrive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from src.models.config import MaskedDiffusionModelConfig, PPIAssetConfig
from src.utils.ppi_asset import PPIAssetMetadata, load_ppi_assets


class PPIAssets(nn.Module):
    """Frozen gene-indexed PPI tables shared by every block of one backbone.

    Attributes exposed to blocks:

    ``spherical_embedding``
        ``[G,r]`` float32, every row on the unit sphere.
    ``expert_ids`` / ``expert_weights``
        ``[G,k]`` int64 / float32.  Rows are ascending by expert id, so column
        zero is *not* the dominant expert; the weights carry that information.
    ``free_mask``
        ``[G]`` bool, True where the spherical embedding is a free (ballast)
        direction rather than a fitted one.
    ``prototypes``
        ``[E,r]`` float32 unit-norm expert prototypes, diagnostic only.
    ``free_mask_float``
        ``[G]`` float32 view of ``free_mask``, derived, read per chunk by the
        PPI attention's norm correction.
    ``gated_embedding``
        ``[G,r]`` float32, ``spherical_embedding`` with every free row set to
        zero.  Derived, not stored.  The free rows are unit vectors in the
        artifact exactly like the fitted ones, so the zeroing is a deliberate
        override rather than something the data already encodes: without it a
        free gene would contribute full-magnitude spurious similarity (its
        self-similarity is exactly 1.0).  The PPI attention reads this, never
        ``spherical_embedding`` directly.
    """

    def __init__(self, config: PPIAssetConfig, tensors: Dict[str, Tensor]) -> None:
        super().__init__()
        self.config = config
        self.num_genes = config.num_genes
        self.embedding_rank = config.embedding_rank
        self.num_experts = config.num_experts
        self.route_top_k = config.route_top_k
        self.asset_metadata: Optional[PPIAssetMetadata] = None

        self.register_buffer(
            "spherical_embedding",
            tensors["spherical_embedding"].detach().contiguous().clone(),
            persistent=True,
        )
        self.register_buffer(
            "expert_ids",
            tensors["expert_ids"].detach().contiguous().clone(),
            persistent=True,
        )
        self.register_buffer(
            "expert_weights",
            tensors["expert_weights"].detach().contiguous().clone(),
            persistent=True,
        )
        self.register_buffer(
            "free_mask",
            tensors["free_mask"].detach().contiguous().clone(),
            persistent=True,
        )
        self.register_buffer(
            "prototypes",
            tensors["prototypes"].detach().contiguous().clone(),
            persistent=True,
        )

        self._register_derived()
        self.register_load_state_dict_post_hook(self._rebuild_derived_after_load)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: PPIAssetConfig,
        *,
        project_root: Optional[Path] = None,
    ) -> "PPIAssets":
        """Load and audit both artifacts from disk.  Used at training time only."""

        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        tensors, metadata = load_ppi_assets(
            config.embedding_path,
            config.embedding_manifest_path,
            config.routing_path,
            config.routing_manifest_path,
            num_genes=config.num_genes,
            embedding_rank=config.embedding_rank,
            num_experts=config.num_experts,
            route_top_k=config.route_top_k,
            verify_sha256=config.verify_sha256,
            project_root=project_root,
        )
        assets = cls(config, tensors)
        assets.asset_metadata = metadata
        return assets

    @classmethod
    def empty(cls, config: PPIAssetConfig) -> "PPIAssets":
        """Build correctly shaped placeholders for a strict state-dict load.

        The values are meaningless until ``load_state_dict`` overwrites them;
        the post hook then rebuilds the derived routing index.
        """

        tensors = {
            "spherical_embedding": torch.zeros(
                config.num_genes,
                config.embedding_rank,
                dtype=torch.float32,
            ),
            "expert_ids": torch.zeros(
                config.num_genes,
                config.route_top_k,
                dtype=torch.int64,
            ),
            "expert_weights": torch.zeros(
                config.num_genes,
                config.route_top_k,
                dtype=torch.float32,
            ),
            "free_mask": torch.zeros(config.num_genes, dtype=torch.bool),
            "prototypes": torch.zeros(
                config.num_experts,
                config.embedding_rank,
                dtype=torch.float32,
            ),
        }
        return cls(config, tensors)

    # ------------------------------------------------------------------
    # Derived routing index
    # ------------------------------------------------------------------
    def routing_index(self, expert: int) -> Tensor:
        """Gene indices routed to ``expert``, as an int64 ``[n_e]`` tensor."""

        return getattr(self, f"route_index_{expert}")

    def routing_weight(self, expert: int) -> Tensor:
        """Routing weights aligned with :meth:`routing_index`, float32 ``[n_e]``."""

        return getattr(self, f"route_weight_{expert}")

    def _register_derived(self) -> None:
        indices, weights = self._compute_routing_index()
        for expert in range(self.num_experts):
            self.register_buffer(
                f"route_index_{expert}",
                indices[expert],
                persistent=False,
            )
            self.register_buffer(
                f"route_weight_{expert}",
                weights[expert],
                persistent=False,
            )
        self.register_buffer(
            "gated_embedding",
            self._compute_gated_embedding(),
            persistent=False,
        )
        # The attention needs the free rows as a float multiplier on every
        # chunk; deriving it once here keeps that out of the hot path.
        self.register_buffer(
            "free_mask_float",
            self.free_mask.to(torch.float32),
            persistent=False,
        )

    def _compute_gated_embedding(self) -> Tensor:
        """Spherical embedding with the free (ballast) rows zeroed."""

        return self.spherical_embedding.masked_fill(
            self.free_mask.unsqueeze(1),
            0.0,
        ).contiguous()

    def _compute_routing_index(self):
        expert_ids = self.expert_ids
        expert_weights = self.expert_weights
        device = expert_ids.device

        gene_axis = torch.arange(self.num_genes, dtype=torch.int64, device=device)
        flat_genes = gene_axis.repeat_interleave(self.route_top_k)
        flat_experts = expert_ids.reshape(-1)
        flat_weights = expert_weights.reshape(-1)

        indices = []
        weights = []
        for expert in range(self.num_experts):
            selected = flat_experts == expert
            indices.append(flat_genes[selected].contiguous())
            weights.append(flat_weights[selected].to(torch.float32).contiguous())
        return indices, weights

    @staticmethod
    def _rebuild_derived_after_load(module: "PPIAssets", incompatible_keys) -> None:
        """Rebuild every derived buffer from the tables that were just loaded."""

        indices, weights = module._compute_routing_index()
        for expert in range(module.num_experts):
            setattr(module, f"route_index_{expert}", indices[expert])
            setattr(module, f"route_weight_{expert}", weights[expert])
        module.gated_embedding = module._compute_gated_embedding()
        module.free_mask_float = module.free_mask.to(torch.float32)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"num_genes={self.num_genes}, embedding_rank={self.embedding_rank}, "
            f"num_experts={self.num_experts}, route_top_k={self.route_top_k}"
        )

    def forward(self) -> None:  # pragma: no cover - container module
        raise RuntimeError(
            "PPIAssets is a buffer container; read its tensors directly."
        )


def build_ppi_assets(
    config: MaskedDiffusionModelConfig,
    *,
    load_from_disk: bool,
    project_root: Optional[Path] = None,
) -> Optional[PPIAssets]:
    """Return the shared assets a configuration needs, or ``None`` for Performer.

    ``load_from_disk`` selects the training path (read and audit the artifacts)
    or the inference path (shaped placeholders filled by a strict load).
    """

    if config.ppi is None:
        return None
    if load_from_disk:
        return PPIAssets.from_config(config.ppi, project_root=project_root)
    return PPIAssets.empty(config.ppi)


__all__ = ["PPIAssets", "build_ppi_assets"]
