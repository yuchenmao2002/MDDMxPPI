"""Ordered same-resolution denoising backbone."""

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
    """Stack arbitrary blocks that honor the stable ``DenoiserBlock`` API.

    The backbone owns only ordered execution, optional activation checkpointing,
    namespacing of auxiliary outputs, and one final ``LayerNorm(d_model)``.  It
    must not branch on concrete block types during ``forward``.  Future blocks
    may therefore be mixed with Performer as long as each restores
    ``[B,19295,d_model]`` at its boundary.
    """

    def __init__(
        self,
        blocks: Sequence[DenoiserBlock],
        *,
        d_model: int,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if not blocks:
            raise ValueError("DenoiserBackbone requires at least one block.")
        if any(not isinstance(block, DenoiserBlock) for block in blocks):
            raise TypeError("Every backbone element must implement DenoiserBlock.")

        self.blocks = nn.ModuleList(blocks)
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: Tensor,
        context: DenoiserContext,
        *,
        output_hidden_states: bool = False,
        return_diagnostics: bool = False,
    ) -> BackboneOutput:
        """Run the ordered block stack and final normalization.

        Intermediate ``[B,G,d]`` tensors are returned only when explicitly
        requested because retaining all L layers is prohibitively expensive at
        19,295 tokens.
        """

        self._validate_inputs(hidden_states, context)
        expected_shape = hidden_states.shape
        expected_dtype = hidden_states.dtype
        expected_device = hidden_states.device

        hidden_history = [hidden_states] if output_hidden_states else None
        aux_losses = {}
        diagnostics = {} if return_diagnostics else None

        for block_index, block in enumerate(self.blocks):
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # Bind the current block as a default argument: a closure over
                # the loop variable would otherwise point at the last block
                # when checkpoint recomputation happens during backward.
                def run_block(
                    states: Tensor,
                    current_block: DenoiserBlock = block,
                ) -> BlockOutput:
                    return current_block(
                        states,
                        context,
                        return_diagnostics=return_diagnostics,
                    )

                # Non-reentrant checkpointing supports the structured
                # BlockOutput contract, allows parameter gradients even when
                # the input itself does not require grad, and preserves dropout
                # RNG state for recomputation.
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
                self._validate_output_name(name, kind="auxiliary loss")
                aux_losses[prefix + name] = value

            if return_diagnostics:
                block_diagnostics = block_output.diagnostics or {}
                for name, value in block_diagnostics.items():
                    self._validate_output_name(name, kind="diagnostic")
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

        last_hidden_state = self.final_norm(hidden_states)
        if hidden_history is not None:
            # The last element represents the public output of the final block,
            # including the backbone-owned final normalization.  This keeps
            # hidden_states[-1] identical to last_hidden_state without retaining
            # an otherwise redundant pre-normalization copy.
            hidden_history[-1] = last_hidden_state

        return BackboneOutput(
            last_hidden_state=last_hidden_state,
            aux_losses=aux_losses,
            diagnostics=diagnostics,
            hidden_states=(
                tuple(hidden_history) if hidden_history is not None else None
            ),
        )

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        context: DenoiserContext,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, genes, d_model].")
        batch_size, num_genes, width = hidden_states.shape
        if batch_size <= 0 or num_genes <= 0:
            raise ValueError("batch and gene dimensions must both be non-zero.")
        if width != self.d_model:
            raise ValueError(f"Expected hidden width {self.d_model}, got {width}.")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point.")
        if context.diffusion_time.shape != (batch_size,):
            raise ValueError(
                "diffusion_time must have shape [batch]; got "
                f"{tuple(context.diffusion_time.shape)}."
            )
        if context.diffusion_time.dtype != torch.float32:
            raise TypeError("diffusion_time must have dtype float32.")
        if context.diffusion_mask.shape != (batch_size, num_genes):
            raise ValueError(
                "diffusion_mask must have shape [batch, genes]; got "
                f"{tuple(context.diffusion_mask.shape)}."
            )
        if context.diffusion_mask.dtype != torch.bool:
            raise TypeError("diffusion_mask must have dtype bool.")
        if context.diffusion_time.device != hidden_states.device:
            raise ValueError("diffusion_time and hidden_states must share a device.")
        if context.diffusion_mask.device != hidden_states.device:
            raise ValueError("diffusion_mask and hidden_states must share a device.")

    @staticmethod
    def _validate_block_output(
        block_output: BlockOutput,
        *,
        expected_shape: torch.Size,
        expected_dtype: torch.dtype,
        expected_device: torch.device,
        block_index: int,
    ) -> None:
        if not isinstance(block_output, BlockOutput):
            raise TypeError(
                f"Block {block_index} must return BlockOutput, got "
                f"{type(block_output).__name__}."
            )
        hidden_states = block_output.hidden_states
        if not isinstance(hidden_states, Tensor):
            raise TypeError(f"Block {block_index} hidden_states must be a Tensor.")
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
        if not isinstance(block_output.aux_losses, dict):
            raise TypeError(f"Block {block_index} aux_losses must be a dict.")
        for name, value in block_output.aux_losses.items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Auxiliary loss blocks.{block_index}/{name} must be a Tensor."
                )
            if value.ndim != 0 or value.dtype != torch.float32:
                raise ValueError(
                    f"Auxiliary loss blocks.{block_index}/{name} must be an "
                    "FP32 scalar."
                )
            if value.device != expected_device:
                raise ValueError(
                    f"Auxiliary loss blocks.{block_index}/{name} must be on "
                    f"{expected_device}, got {value.device}."
                )
        if block_output.diagnostics is not None and not isinstance(
            block_output.diagnostics, dict
        ):
            raise TypeError(f"Block {block_index} diagnostics must be a dict or None.")

    @staticmethod
    def _validate_output_name(name: str, *, kind: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Every {kind} name must be a non-empty string.")


def build_performer_backbone(config: PerformerConfig) -> DenoiserBackbone:
    """Construct ``config.num_layers`` independent standard Performer blocks."""

    blocks = [
        PerformerBlock(config, layer_index=layer_index)
        for layer_index in range(config.num_layers)
    ]
    return DenoiserBackbone(
        blocks,
        d_model=config.d_model,
        activation_checkpointing=config.activation_checkpointing,
    )
