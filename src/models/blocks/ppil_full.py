"""PPIL full block: PPI linear attention together with the routed feed-forward.

One of four interchangeable backbone variants.  It installs both new components
into the pre-normalized residual shell shared with the other PPIL blocks, so
the variants differ only where they are meant to differ.
"""

from __future__ import annotations

from src.models.blocks.ppil_components import (
    PPILBlockBase,
    PPILSelfAttention,
    PPIStaticMoEFeedForward,
)
from src.models.config import PPILFullConfig
from src.models.ppi_assets import PPIAssets


class PPILFullBlock(PPILBlockBase):
    """PPI linear attention + statically routed mixture-of-experts feed-forward."""

    def __init__(
        self,
        config: PPILFullConfig,
        assets: PPIAssets,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()

        self.layer_index = layer_index
        # ``mixer`` is the stable public name shared with every block type.
        self.mixer = PPILSelfAttention(config, assets, layer_index=layer_index)
        self.ffn = PPIStaticMoEFeedForward(config, assets, layer_index=layer_index)
        self._install_norms(
            config,
            mixer_takes_gate=True,
            ffn_takes_diagnostics=True,
        )


__all__ = ["PPILFullBlock"]
