"""
Continuous-time Absorbing-mask Forward Process
Learnable MASK state
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Generator, Tensor, nn

from src.models.config import ForwardProcessConfig
from src.models.types import ForwardProcessOutput
from src.utils.tensor_validation import validate_diffusion_time



class AbsorbingMaskForwardProcess:
    """
    采样前向过程中的边缘噪声分布
    For each cell, sample t_b ~ Uniform[0,1) independently.
    Conditional on that time, sample genes independently as M_i ~ Bernoulli(t).
    Do not force at least one mask. 
    Explicit caller-supplied times may include 1.0 so all-MASK validation and the future sampler can represent the endpoint.
    This object owns no learnable state and must not sample implicitly inside the core denoiser.
    """

    def __init__(self, config: ForwardProcessConfig) -> None:
        self.config = config


    def sample_times(self, batch_size: int, *, device: torch.device, generator: Optional[Generator] = None) -> Tensor:
        """
        时间采样
        为 batch 中每一个样本独立生成一个范围在 [0,1) 连续区间的时间 FP32 [B]
        """

        return torch.rand(
            (batch_size,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )


    def sample_mask(self, diffusion_time: Tensor, *, generator: Optional[Generator] = None) -> Tensor:
        """
        掩码采样
        Output:
        布尔掩码 [B,G]
        对每个基因独立地进行伯努利抽样
        """

        uniform_draws = torch.rand(
            (diffusion_time.shape[0], self.config.num_genes),
            device=diffusion_time.device,
            dtype=torch.float32,
            generator=generator,
        )
        diffusion_mask = uniform_draws < diffusion_time[:, None]

        return diffusion_mask


    def sample(self, batch_size: int, *, device: torch.device, diffusion_time: Optional[Tensor] = None, generator: Optional[Generator] = None) -> ForwardProcessOutput:
        """
        Output:
        扩散时间和扩散掩码
        允许调用者传入自定义的扩散时间
        """

        expected_device = torch.device(device)
        # 没有传入扩散时间则自行生成
        if diffusion_time is None:
            diffusion_time = self.sample_times(
                batch_size,
                device=expected_device,
                generator=generator,
            )
        else:
            validate_diffusion_time(diffusion_time, batch_size=batch_size)
            if diffusion_time.device != expected_device:
                raise ValueError(
                    "diffusion_time and requested sample device must match; got "
                    f"{diffusion_time.device} and {expected_device}."
                )

        diffusion_mask = self.sample_mask(diffusion_time, generator=generator)
        return ForwardProcessOutput(
            diffusion_time=diffusion_time,
            diffusion_mask=diffusion_mask,
        )



class AbsorbingStateEmbedding(nn.Module):
    """
    根据布尔掩码，将掩码位置替换为可学习的掩码向量 [d]
    所有基因共享
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()

        # 创建掩码向量 [d]
        self.d_model = d_model
        self.mask_embedding = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)


    def forward(self, encoded_expression: Tensor, diffusion_mask: Tensor) -> Tensor:
        """
        Input: 
        原始的基因表达值嵌入 [B,G,d]
        布尔掩码向量 [B,G]
        Output:
        掩码后的基因表达值嵌入 [B,G,d]
        unchanged shape/dtype/device
        """

        mask_embedding = self.mask_embedding.to(dtype=encoded_expression.dtype)
        return torch.where(
            diffusion_mask.unsqueeze(-1),
            mask_embedding.view(1, 1, self.d_model),
            encoded_expression,
        )
