"""PPIL attention-only block: new mixer, baseline dense feed-forward.

The feed-forward is shape-identical to the Performer's, so a comparison against
``ppil_full`` isolates the routed mixture of experts, and a comparison against
``performer`` isolates the PPI attention kernel.  This variant reads only the
spherical embedding; the routing table is unused but still travels with the
shared asset module.
"""

from __future__ import annotations

from src.models.blocks.ppil_components import (
    PPILBlockBase,
    PPILSelfAttention,
    dense_feed_forward,
)
from src.models.config import PPILAttentionOnlyConfig
from src.models.ppi_assets import PPIAssets


class PPILAttentionOnlyBlock(PPILBlockBase):
    """PPI linear attention + the baseline ``Linear(d,4d) -> GELU -> Linear(4d,d)``."""

    def __init__(
        self,
        config: PPILAttentionOnlyConfig,
        assets: PPIAssets,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()

        self.layer_index = layer_index
        # ``mixer`` is the stable public name shared with every block type.
        self.mixer = PPILSelfAttention(config, assets, layer_index=layer_index)
        self.ffn = dense_feed_forward(config)
        self._install_norms(
            config,
            mixer_takes_gate=True,
            ffn_takes_diagnostics=False,
        )


__all__ = ["PPILAttentionOnlyBlock"]
