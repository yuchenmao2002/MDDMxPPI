"""
Likelihood Objectives for Continuous-time Absorbing Expression Diffusion
The primary objective is a single, token-level hurdle NLL.
A token is either exactly zero or positive.
The zero event is modeled by a Bernoulli gate and positive values by a zero-truncated Normal distribution.
Only diffusion-masked tokens contribute, and the continuous-time absorbing schedule alpha(t) = 1 - t contributes its exact 1 / t weight.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.config import LossConfig
from src.models.types import (
    HurdleDistributionParameters,
    TimeWeightedHurdleNLLOutput,
)


_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_PI_OVER_TWO = 0.5 * math.log(math.pi / 2.0)
_SQRT_TWO = math.sqrt(2.0)



def _zero_truncated_normal_nll(target: Tensor, location: Tensor, scale: Tensor) -> Tensor:
    r"""
    零截尾正态分布损失
    返回逐基因的 -log f_+(x; mu, sigma)，其中
    f_+(x; mu, sigma) = phi((x - mu) / sigma) / (sigma * Phi(mu / sigma))，x > 0，
    即类 docstring 中 c_i = 1 分支里的密度项。
    Directly evaluating
    .. math::
       \tfrac12(y-z)^2 + \log\sigma + \tfrac12\log(2\pi)
       + \log\Phi(z),\quad y=x/\sigma,\ z=\mu/\sigma,
    loses precision when z is very negative: its positive and negative
    ``z**2 / 2`` terms are individually huge.  For z < 0 this function uses
    the exact identity
    .. math::
       \Phi(z)=\tfrac12\exp(-z^2/2)
       \operatorname{erfcx}(-z/\sqrt2)
    and analytically cancels those terms before floating-point evaluation:
    .. math::
       \tfrac12y^2-yz+\log\sigma+\tfrac12\log(\pi/2)
       +\log\operatorname{erfcx}(-z/\sqrt2).
    The ordinary log_ndtr expression remains well-conditioned for z >= 0.
    Safe placeholder values keep both vectorized branches finite before where selects the mathematically applicable result.
    """

    # 严格要求输入的目标值、均值和标准差为 FP32
    if target.dtype != torch.float32:
        raise TypeError("target must be FP32 inside truncated-Normal NLL.")
    if location.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError("location and scale must be FP32 inside the NLL.")

    # 对目标值和均值进行标准化 Z-score
    standardized_target = target / scale
    standardized_location = location / scale
    negative_tail = standardized_location < 0.0  # 判断均值是否在 0 以左

    # 负均值分支处理
    tail_location = torch.where(
        negative_tail,
        standardized_location,
        torch.zeros_like(standardized_location),
    )
    tail_nll = (
        0.5 * standardized_target.square()
        - standardized_target * tail_location
        + torch.log(scale)
        + _HALF_LOG_PI_OVER_TWO
        + torch.log(torch.special.erfcx(-tail_location / _SQRT_TWO))
    )

    # 正均值分支处理
    body_location = torch.where(
        negative_tail,
        torch.zeros_like(standardized_location),
        standardized_location,
    )

    # 合并
    body_nll = (
        0.5 * (standardized_target - body_location).square()
        + torch.log(scale)
        + _HALF_LOG_TWO_PI
        + torch.special.log_ndtr(body_location)
    )
    return torch.where(negative_tail, tail_nll, body_nll)


class TimeWeightedHurdleNLLLoss(nn.Module):
    r"""
    计算时间加权的 Hurdle NLL 损失（逐基因表述）

    解码器为每个基因 i 输出三个通道 (eta, mu, sigma)。基因 i 的条件分布
    ``p_theta(x_0^i | x_t, t)`` 由零点质量与截断到 (0, inf) 的 Gaussian 组成：

    .. math::
       (1 - \pi_t^i) \quad\text{和}\quad \pi_t^i f_+(x_0^i; \mu_t^i, \sigma_t^i),
       \qquad \pi_t^i = \operatorname{sigmoid}(\eta_t^i),

    .. math::
       f_+(x; \mu, \sigma) = \frac{\phi((x - \mu) / \sigma)}{\sigma\,\Phi(\mu / \sigma)},
       \qquad x > 0,

    其中 phi 与 Phi 分别是标准 Gaussian 的 PDF 与 CDF。
    令 ``c_i = 1[x_0^i > 0]`` 表示目标基因是否为观测正值，逐基因 hurdle 负对数密度为

    .. math::
       \ell_t^i = (1 - c_i)\operatorname{softplus}(\eta_t^i)
       + c_i\left[\operatorname{softplus}(-\eta_t^i)
       - \log f_+(x_0^i; \mu_t^i, \sigma_t^i)\right].

    注意两个分支的 logit 符号相反：``softplus(eta) = -log(1 - pi)`` 是零事件的代价，
    ``softplus(-eta) = -log pi`` 是正事件的代价。

    对线性吸收 schedule ``alpha(t) = 1 - t``，训练目标为

    .. math::
       \mathcal L_{\mathrm{rec}}(\theta) = \mathbb E_{x_0, t, m_t}
       \left[\frac{1}{tG}\sum_{i=1}^{G} m_{t,i}\,\ell_{t,i}\right].

    本方法返回该期望的批均值估计：分子是 ``sum_{b,i} M_bi t_b^{-1} ell_bi``，
    分母是固定的 cell-gene 位置数 ``B*G``，两者相除即
    ``mean_b[(1/(t_b G)) sum_i m_bi ell_bi]``。分母**恒为固定的位置数**，
    绝不使用随机的 masked token 计数。
    ``t=0`` 的行只在其没有任何 masked 位置时合法，并贡献一个可微的零。

    返回的各项和是局部充分统计量。注意 ``weighted_zero_nll_sum`` 与
    ``weighted_positive_nll_sum`` 的切分是 **c=0 分支与 c=1 分支**，
    而非「检测项与密度项」——正分支中 ``softplus(-eta)`` 与 ``-log f_+`` 是合并上报的。
    在 DDP 下，trainer 必须对 detach 后的充分统计量做全局归约，并以全局固定的
    normalizer 缩放各 rank 的可微分子；只有当每个 rank 的 cell 数相同时，
    独立平均各 rank 的 loss 才等价。
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config


    def forward(self, distribution_parameters: HurdleDistributionParameters, target: Tensor, diffusion_time: Tensor, diffusion_mask: Tensor) -> TimeWeightedHurdleNLLOutput:
        """
        Score hurdle parameters against clean target expression.
        Args:
            distribution_parameters: Strongly typed decoder tensors
                detection_logits, positive_location and positive_scale, each shaped [B, G, 1].
                Positive scales must already include the decoder's minimum-scale floor.
            target: Finite, non-negative clean expression [B, 19295, 1].
            diffusion_time: FP32 continuous times [B] in [0, 1].
            diffusion_mask: Boolean [B, G]; True means absorbing MASK and is the only kind of position scored by this loss.
        Returns:
            FP32 differentiable loss/sums plus integer sufficient statistics.
        """

        batch_size = target.shape[0]

        detection_logits = distribution_parameters.detection_logits
        positive_location = distribution_parameters.positive_location
        positive_scale = distribution_parameters.positive_scale

        # 构建掩码
        # 掩码位置分为真实值为 0 的位置和真实值大于 0 的位置
        is_positive = target > 0.0
        expanded_mask = diffusion_mask.unsqueeze(-1)
        masked_zero = expanded_mask & ~is_positive
        masked_positive = expanded_mask & is_positive

        # No epsilon or clipping is permitted in 1/t: doing so would change the
        # objective.  The safe branch solely defines valid, unmasked t=0 rows as
        # zero contribution without ever evaluating a reciprocal at zero.
        # 计算时间权重
        # t = 0 不贡献损失
        positive_time = diffusion_time > 0.0
        safe_time = torch.where(
            positive_time,
            diffusion_time,
            torch.ones_like(diffusion_time),
        )
        inverse_time = torch.where(
            positive_time,
            safe_time.reciprocal(),
            torch.zeros_like(safe_time),
        ).view(batch_size, 1, 1)

        # Evaluate each likelihood branch only on the tokens that use it.  In
        # addition to avoiding wasted work on a very long gene axis, this keeps
        # an irrelevant extreme positive-distribution parameter at a zero target
        # from producing ``0 * inf -> NaN`` in the zero branch.
        # 目标值为0的 NLL 计算
        expanded_inverse_time = inverse_time.expand_as(target)
        zero_logits = detection_logits.masked_select(masked_zero)
        zero_weights = expanded_inverse_time.masked_select(masked_zero)
        weighted_zero_nll_sum = (
            F.softplus(zero_logits) * zero_weights
        ).sum(dtype=torch.float32)

        # 目标值>0的 NLL 计算
        positive_logits = detection_logits.masked_select(masked_positive)
        positive_target = target.masked_select(masked_positive)
        positive_location_selected = positive_location.masked_select(masked_positive)
        positive_scale_selected = positive_scale.masked_select(masked_positive)
        positive_weights = expanded_inverse_time.masked_select(masked_positive)
        positive_value_nll = _zero_truncated_normal_nll(
            positive_target,
            positive_location_selected,
            positive_scale_selected,
        )
        positive_nll = F.softplus(-positive_logits) + positive_value_nll
        weighted_positive_nll_sum = (
            positive_nll * positive_weights
        ).sum(dtype=torch.float32)
        # Define the total from its two reported components so the accounting
        # identity is exact, including for empty selections.
        # 统计指标与最终 Loss 计算
        weighted_nll_sum = weighted_zero_nll_sum + weighted_positive_nll_sum

        cell_count = torch.tensor(
            batch_size,
            dtype=torch.int64,
            device=target.device,
        )
        # 归一化常数是固定的 cell-gene 位置数 B*G，从张量形状推导而非取模块常量：
        # 公式里的 G 就是本次输入的基因数，写死 NUM_GENES 会让任何其它 G 静默地按
        # 错误的分母归一而不报错。
        normalizer = torch.tensor(
            batch_size * target.shape[1],
            dtype=torch.int64,
            device=target.device,
        )
        masked_count = diffusion_mask.sum(dtype=torch.int64)
        masked_zero_count = masked_zero.sum(dtype=torch.int64)
        masked_positive_count = masked_positive.sum(dtype=torch.int64)

        loss = weighted_nll_sum / normalizer.to(dtype=torch.float32)

        return TimeWeightedHurdleNLLOutput(
            loss=loss,
            weighted_nll_sum=weighted_nll_sum,
            normalizer=normalizer,
            cell_count=cell_count,
            masked_count=masked_count,
            masked_zero_count=masked_zero_count,
            masked_positive_count=masked_positive_count,
            weighted_zero_nll_sum=weighted_zero_nll_sum,
            weighted_positive_nll_sum=weighted_positive_nll_sum,
        )
