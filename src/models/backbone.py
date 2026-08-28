"""Ordered Same-resolution Blocks"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.models.blocks.base import DenoiserBlock
from src.models.blocks.performer import PerformerBlock
from src.models.config import PerformerConfig
from src.models.types import BackboneOutput, BlockOutput, DenoiserContext


class DenoiserBackbone(nn.Module):
    """
    有序去噪网络主干
    Stack arbitrary blocks that honor the stable DenoiserBlock API.
    The backbone owns only ordered execution, optional activation checkpointing, namespacing of auxiliary outputs, and one final LayerNorm(d).
    It must not branch on concrete block types during forward.
    Blocks may therefore be mixed as long as each restores [B,19295,d] at its boundary.
    """

    def __init__(self, blocks: Sequence[DenoiserBlock], *, d_model: int, activation_checkpointing: bool = False) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        self.final_norm = nn.LayerNorm(d_model)


    def forward(self, hidden_states: Tensor, context: DenoiserContext, *, output_hidden_states: bool = False, return_diagnostics: bool = False) -> BackboneOutput:
        """将 Blocks 有序堆叠并归一化"""

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
