"""
Training-time Assembly for Masked Expression Diffusion
The deterministic denoiser consumes an explicitly specified state.
This thin training wrapper owns random forward corruption and the time-weighted hurdle-density objective.
"""

from __future__ import annotations

from typing import Optional

from torch import Generator, Tensor, nn

from src.models.config import MaskedDiffusionModelConfig
from src.models.losses import TimeWeightedHurdleNLLLoss
from src.models.masked_expression_denoiser import MaskedExpressionDenoiser
from src.models.masking import AbsorbingMaskForwardProcess
from src.models.types import TrainingOutput
from src.utils.tensor_validation import (
    validate_diffusion_mask,
    validate_diffusion_time,
)


class MaskedDiffusionTrainingModule(nn.Module):
    """
    Training-time wrapper around the deterministic denoiser.
    If time and mask are supplied, they are used verbatim after validation.
    If neither is supplied, this wrapper samples per-cell Uniform times and conditionally independent masks.
    Passing a mask without its matching time is invalid.
    The clean expression tensor is both the visible-state source and the loss target, but masked values are removed inside the denoiser before any cross-gene operation.
    """

    def __init__(self, denoiser: MaskedExpressionDenoiser, forward_process: AbsorbingMaskForwardProcess, reconstruction_loss: TimeWeightedHurdleNLLLoss) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.forward_process = forward_process
        self.reconstruction_loss = reconstruction_loss


    @classmethod
    def from_config(cls, config: MaskedDiffusionModelConfig) -> "MaskedDiffusionTrainingModule":
        """Assemble forward process, denoiser and objective from one config."""

        return cls(
            denoiser=MaskedExpressionDenoiser.from_config(config),
            forward_process=AbsorbingMaskForwardProcess(config.forward_process),
            reconstruction_loss=TimeWeightedHurdleNLLLoss(config.loss),
        )


    def forward(self, clean_expression: Tensor, *, diffusion_time: Optional[Tensor] = None, diffusion_mask: Optional[Tensor] = None, generator: Optional[Generator] = None, return_diagnostics: bool = False) -> TrainingOutput:
        """
        Corrupt, denoise and locally score a clean training batch.
        The hurdle NLL is evaluated only at diffusion-masked positions, weighted by 1/t
        and normalized by the fixed local B*G.
        Returned weighted sums and counts are local sufficient statistics.
        """

        batch_size = clean_expression.shape[0]
        if diffusion_mask is not None and diffusion_time is None:
            raise ValueError(
                "diffusion_time is required when diffusion_mask is supplied."
            )

        # 逻辑分支 1
        # 没有提供掩码（正常训练情况）
        if diffusion_mask is None:
            forward_state = self.forward_process.sample(
                batch_size,
                device=clean_expression.device,
                diffusion_time=diffusion_time,
                generator=generator,
            )
            diffusion_time = forward_state.diffusion_time
            diffusion_mask = forward_state.diffusion_mask
        # 逻辑分支 2
        # 提供了掩码（用于特定测试或固定状态训练）
        else:
            # Both are non-None because the mask-without-time case is rejected.
            validate_diffusion_time(diffusion_time, batch_size=batch_size)
            validate_diffusion_mask(
                diffusion_mask,
                batch_size=batch_size,
                num_genes=self.denoiser.num_genes,
            )
            if diffusion_time.device != clean_expression.device:
                raise ValueError(
                    "diffusion_time and clean_expression must share a device."
                )
            if diffusion_mask.device != clean_expression.device:
                raise ValueError(
                    "diffusion_mask and clean_expression must share a device."
                )
            zero_time_rows = diffusion_time == 0.0
            if (
                zero_time_rows.any().item()
                and diffusion_mask[zero_time_rows].any().item()
            ):
                raise ValueError("Rows at t=0 must contain no diffusion MASK states.")
            one_time_rows = diffusion_time == 1.0
            if (
                one_time_rows.any().item()
                and (~diffusion_mask[one_time_rows]).any().item()
            ):
                raise ValueError("Rows at t=1 must be entirely diffusion-masked.")

        # 将干净的数据、扩散时间和掩码传入去噪器
        model_output = self.denoiser(
            clean_expression,
            diffusion_time,
            diffusion_mask,
            return_hidden_state=False,
            output_hidden_states=False,
            return_diagnostics=return_diagnostics,
            compute_point_prediction=False,
        )

        # 计算损失
        parameters = model_output.decoder_output.distribution_parameters
        reconstruction = self.reconstruction_loss(
            parameters,
            clean_expression,
            diffusion_time,
            diffusion_mask,
        )

        return TrainingOutput(
            loss=reconstruction.loss,
            reconstruction_loss=reconstruction.loss,
            weighted_nll_sum=reconstruction.weighted_nll_sum,
            normalizer=reconstruction.normalizer,
            cell_count=reconstruction.cell_count,
            masked_count=reconstruction.masked_count,
            masked_zero_count=reconstruction.masked_zero_count,
            masked_positive_count=reconstruction.masked_positive_count,
            weighted_zero_nll_sum=reconstruction.weighted_zero_nll_sum,
            weighted_positive_nll_sum=(
                reconstruction.weighted_positive_nll_sum
            ),
            # The training/NLL path intentionally never computes the hurdle distribution mean.
            # Direct denoiser calls retain the default inference behavior and return it.
            prediction=None,
            diffusion_time=diffusion_time,
            diffusion_mask=diffusion_mask,
            aux_losses=model_output.aux_losses,
            diagnostics=model_output.diagnostics,
        )
