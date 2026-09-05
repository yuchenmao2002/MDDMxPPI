"""PPIL feed-forward-only block: baseline FAVOR+ mixer, routed feed-forward.

The mixer is the unmodified ``PerformerSelfAttention`` imported from
``blocks/performer.py`` — not a copy of it.  ``PPILFeedForwardOnlyConfig``
subclasses ``PerformerConfig``, so every attribute that class reads is present
and the baseline attention needs no change at all.

This variant reads only the routing table; the spherical embedding is unused
but still travels with the shared asset module.
"""

from __future__ import annotations

from src.models.blocks.performer import PerformerSelfAttention
from src.models.blocks.ppil_components import PPILBlockBase, PPIStaticMoEFeedForward
from src.models.config import PPILFeedForwardOnlyConfig
from src.models.ppi_assets import PPIAssets


class PPILFeedForwardOnlyBlock(PPILBlockBase):
    """Baseline FAVOR+ attention + statically routed mixture-of-experts feed-forward."""

    def __init__(
        self,
        config: PPILFeedForwardOnlyConfig,
        assets: PPIAssets,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()

        self.layer_index = layer_index
        # ``mixer`` is the stable public name shared with every block type.
        self.mixer = PerformerSelfAttention(config, layer_index=layer_index)
        self.ffn = PPIStaticMoEFeedForward(config, assets, layer_index=layer_index)
        self._install_norms(
            config,
            mixer_takes_gate=False,
            ffn_takes_diagnostics=True,
        )


__all__ = ["PPILFeedForwardOnlyBlock"]
