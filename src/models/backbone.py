"""Ordered Same-resolution Blocks"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.models.architecture import verify_backbone_matches
from src.models.blocks.base import DenoiserBlock
from src.models.blocks.performer import PerformerBlock
from src.models.blocks.ppil_attention_only import PPILAttentionOnlyBlock
from src.models.blocks.ppil_ffn_only import PPILFeedForwardOnlyBlock
from src.models.blocks.ppil_full import PPILFullBlock
from src.models.config import (
    BACKBONE_VARIANT_PERFORMER,
    MaskedDiffusionModelConfig,
    PerformerConfig,
)
from src.models.ppi_assets import PPIAssets
from src.models.types import BackboneOutput, BlockOutput, DenoiserContext


class DenoiserBackbone(nn.Module):
    """
    有序去噪网络主干
    A homogeneous stack: L layers of one block variant, all honoring the stable DenoiserBlock API.
    The backbone owns only ordered execution, optional activation checkpointing, namespacing of auxiliary outputs, one final LayerNorm(d), and any assets its blocks share.
    It must not branch on concrete block types during forward — that stays true regardless of the stack being homogeneous, and is what keeps new variants free of backbone changes.
    Mixing block types in one stack is not supported; ``build_denoiser_backbone`` rejects it through the architecture-signature assertion.

    ``shared_assets`` is the single owner of tensors every block reads, such as
    the gene-indexed PPI tables.  Blocks hold it by reference without
    registering it, so the state dict contains exactly one copy however many
    layers are stacked, while ``model.to(...)`` still moves it.

    ``num_fourier_bands`` enables the realized-mask-rate features.  They are
    identical for every layer, so the backbone derives them once and hands each
    block an augmented context.  Backbones whose blocks do not read them leave
    this ``None``, and their forward pass is unchanged.
    """

    def __init__(self, blocks: Sequence[DenoiserBlock], *, d_model: int, activation_checkpointing: bool = False, shared_assets: Optional[nn.Module] = None, num_fourier_bands: Optional[int] = None) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        self.final_norm = nn.LayerNorm(d_model)
        self.num_fourier_bands = num_fourier_bands
        if shared_assets is not None:
            self.shared_assets = shared_assets


    def _augment_context(self, context: DenoiserContext) -> DenoiserContext:
        """Derive the realized mask rate and its Fourier representation once.

        ``p_t`` is the fraction of genes still in the absorbing state for each
        cell.  It is deliberately the *realized* rate rather than the supplied
        diffusion time, because that is the quantity the reverse sampler can
        actually observe at every step.  Both outputs are a deterministic
        function of a boolean mask and therefore carry no gradient.
        """

        if self.num_fourier_bands is None:
            return context

        mask_rate = context.diffusion_mask.float().mean(dim=1)  # [B]
        bands = torch.pow(
            2.0,
            torch.arange(
                self.num_fourier_bands,
                dtype=torch.float32,
                device=mask_rate.device,
            ),
        ) * math.pi  # [h]
        phase = mask_rate.unsqueeze(1) * bands.unsqueeze(0)  # [B,h]
        # Interleaved as [sin(band 0), cos(band 0), sin(band 1), ...].
        features = torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(
            mask_rate.shape[0],
            2 * self.num_fourier_bands,
        )
        return replace(
            context,
            mask_rate=mask_rate.detach(),
            mask_rate_features=features.detach(),
        )

    def forward(self, hidden_states: Tensor, context: DenoiserContext, *, output_hidden_states: bool = False, return_diagnostics: bool = False) -> BackboneOutput:
        """将 Blocks 有序堆叠并归一化"""

        context = self._augment_context(context)
        expected_shape = hidden_states.shape
        expected_dtype = hidden_states.dtype
        expected_device = hidden_states.device

        hidden_history = [hidden_states] if output_hidden_states else None  # 用户请求输出中间层状态
        aux_losses = {}                                                     # 收集每一层可能产生的辅助损失
        diagnostics = {} if return_diagnostics else None                    # 诊断信息

        # 遍历网络层
        for block_index, block in enumerate(self.blocks):
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                def run_block(states: Tensor, current_block: DenoiserBlock = block) -> BlockOutput:
                    return current_block(
                        states,
                        context,
                        return_diagnostics=return_diagnostics,
                    )

                block_output = checkpoint(
                    run_block,
                    hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                block_output = block(
                    hidden_states,
                    context,
                    return_diagnostics=return_diagnostics,
                )

            self._validate_block_output(
                block_output,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
                expected_device=expected_device,
                block_index=block_index,
            )
            hidden_states = block_output.hidden_states

            prefix = f"blocks.{block_index}/"
            for name, value in block_output.aux_losses.items():
                aux_losses[prefix + name] = value

            if return_diagnostics:
                block_diagnostics = block_output.diagnostics or {}
                for name, value in block_diagnostics.items():
                    if not isinstance(value, Tensor):
                        raise TypeError(
                            f"Diagnostic {prefix + name!r} must be a Tensor."
                        )
                    if value.requires_grad:
                        raise ValueError(
                            f"Diagnostic {prefix + name!r} must be detached."
                        )
                    assert diagnostics is not None
                    diagnostics[prefix + name] = value

            if hidden_history is not None:
                hidden_history.append(hidden_states)

        # 最终归一化
        last_hidden_state = self.final_norm(hidden_states)

        if hidden_history is not None:
            hidden_history[-1] = last_hidden_state

        return BackboneOutput(
            last_hidden_state=last_hidden_state,
            aux_losses=aux_losses,
            diagnostics=diagnostics,
            hidden_states=(
                tuple(hidden_history) if hidden_history is not None else None
            ),
        )

    
    @staticmethod
    def _validate_block_output(block_output: BlockOutput, *, expected_shape: torch.Size, expected_dtype: torch.dtype, expected_device: torch.device, block_index: int) -> None:

        hidden_states = block_output.hidden_states

        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"Block {block_index} changed hidden shape from "
                f"{tuple(expected_shape)} to {tuple(hidden_states.shape)}."
            )
        if hidden_states.dtype != expected_dtype:
            raise TypeError(
                f"Block {block_index} changed hidden dtype from "
                f"{expected_dtype} to {hidden_states.dtype}."
            )
        if hidden_states.device != expected_device:
            raise ValueError(
                f"Block {block_index} moved hidden states from "
                f"{expected_device} to {hidden_states.device}."
            )



def build_performer_backbone(config: PerformerConfig) -> DenoiserBackbone:
    """
    使用 Performer 的 baseline backbone
    使用 L 个独立的标准 Performer blocks
    """

    blocks = [
        PerformerBlock(config, layer_index=layer_index)
        for layer_index in range(config.num_layers)
    ]
    return DenoiserBackbone(
        blocks,
        d_model=config.d_model,
        activation_checkpointing=config.activation_checkpointing,
    )


# Every PPIL block takes the shared assets as its second positional argument;
# the Performer block predates them and takes none.
_PPI_BLOCK_TYPES = {
    "ppil_attention": PPILAttentionOnlyBlock,
    "ppil_ffn": PPILFeedForwardOnlyBlock,
    "ppil_full": PPILFullBlock,
}


def build_denoiser_backbone(config: MaskedDiffusionModelConfig, *, ppi_assets: Optional[PPIAssets]) -> DenoiserBackbone:
    """
    按变体构建 backbone
    Build ``num_layers`` blocks of the configured variant, hand every block the
    same shared asset object, and verify that the stack really has the
    architecture the configuration claims before returning it.
    """

    backbone_config = config.backbone
    variant = config.backbone_variant

    if variant == BACKBONE_VARIANT_PERFORMER:
        if ppi_assets is not None:
            raise ValueError(
                "The Performer backbone does not read the PPI assets; passing "
                "them would put unused tensors into every checkpoint."
            )
        backbone = build_performer_backbone(backbone_config)
    else:
        block_type = _PPI_BLOCK_TYPES.get(variant)
        if block_type is None:
            raise ValueError(f"Unknown backbone_variant={variant!r}.")
        if ppi_assets is None:
            raise ValueError(
                f"backbone_variant={variant!r} requires the shared PPI assets."
            )
        blocks = [
            block_type(backbone_config, ppi_assets, layer_index=layer_index)
            for layer_index in range(backbone_config.num_layers)
        ]
        # Only the variants whose mixer reads the PPI prior gate need the
        # mask-rate features; one band per attention head.
        needs_gate = hasattr(backbone_config, "ppi_rank")
        backbone = DenoiserBackbone(
            blocks,
            d_model=backbone_config.d_model,
            activation_checkpointing=backbone_config.activation_checkpointing,
            shared_assets=ppi_assets,
            num_fourier_bands=backbone_config.num_heads if needs_gate else None,
        )

    # The configuration's claim about its own architecture is only worth
    # something once it has been checked against what was actually built.
    verify_backbone_matches(
        backbone,
        expected_signature=config.backbone_signature,
        context="Backbone construction",
    )
    return backbone
