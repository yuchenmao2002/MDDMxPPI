"""
Non-causal Softmax FAVOR+ Performer Block
Random-feature maps and all kernel sufficient statistics are accumulated in FP32 even when surrounding projections and residual activations use BF16 autocast.
The random projection matrix is a persistent, non-trainable buffer owned by each attention layer.
It is shared across batch items and heads within that layer.
The current model keeps it fixed for reproducibility; any future redraw is an explicit trainer action synchronized across distributed ranks, never a side effect of forward.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from src.models.blocks.base import DenoiserBlock
from src.models.config import PerformerConfig
from src.models.types import BlockOutput, DenoiserContext



def make_orthogonal_random_matrix(num_rows: int, num_columns: int, *, seed: int, device: Optional[object] = None) -> Tensor:
    """
    生成正交随机特征矩阵
    Use QR-derived ``num_columns x num_columns`` blocks, stack and truncate to ``num_rows``.
    Row lengths follow the chi distribution with ``num_columns`` degrees of freedom (the Performer ``ortho_scaling=0`` convention).
    Construction must be deterministic for ``seed`` and must not mutate the process-global RNG state.
    """

    # 在 CPU 上生成随机数
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    num_blocks = math.ceil(num_rows / num_columns)
    # 生成非结构化高斯矩阵
    unstructured = torch.randn(
        num_blocks,
        num_columns,
        num_columns,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )

    # QR 分解（正交化处理）
    orthogonal, _ = torch.linalg.qr(unstructured, mode="reduced")
    matrix = orthogonal.transpose(-2, -1).reshape(-1, num_columns)[:num_rows]

    # Chi 分布缩放
    # 高斯向量的长度遵循 Chi 分布
    gaussian_rows = torch.randn(
        num_rows,
        num_columns,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    # 对正交行向量重新乘以独立采样的长度，恢复高斯分布的径向特性
    row_lengths = torch.linalg.vector_norm(gaussian_rows, dim=1)
    matrix = matrix * row_lengths.unsqueeze(1)

    if device is not None:
        matrix = matrix.to(device=device)
    return matrix



class PerformerSelfAttention(nn.Module):
    """
    非因果多头线性注意力
    num_heads = 8
    head_dim = 64
    num_random_features = 256
    Q/K are scaled by ``head_dim**(-1/4)`` before the positive feature map so the approximated kernel is ``exp(q @ k / sqrt(head_dim))``.

    Non-causal attention is evaluated as::
        S = K_prime.transpose(-2,-1) @ V
        z = K_prime.sum(sequence_axis)
        output_i = (Q_prime_i @ S) / (Q_prime_i @ z)
    """

    def __init__(self, config: PerformerConfig, *, layer_index: int) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_random_features = config.num_random_features
        self.sequence_chunk_size = config.sequence_chunk_size
        self.feature_epsilon = config.feature_epsilon
        self.layer_index = layer_index

        # Q/K/V 不使用 biases
        self.query_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output_projection = nn.Linear(config.d_model, config.d_model)

        projection_seed = config.projection_seed + layer_index
        projection = make_orthogonal_random_matrix(
            config.num_random_features,
            config.head_dim,
            seed=projection_seed,
        )
        self.register_buffer("projection_matrix", projection, persistent=True)

        self._query_key_scale = config.head_dim**-0.25          ## Q/K 缩放因子
        self._feature_scale = config.num_random_features**-0.5  ## 特征缩放因子


    def forward(self, hidden_states: Tensor) -> Tensor:
        """Input & Output: [B,G,d]"""

        batch_size, sequence_length, _ = hidden_states.shape
        output_dtype = hidden_states.dtype

        # 遍历 1
        # 全局全局键值稳定
        # 找出所有 Key 映射到高斯特征后的全局最大值
        key_max: Optional[Tensor] = None
        with torch.no_grad():
            for start, stop in self._chunk_ranges(sequence_length):
                key_chunk = self._split_heads(
                    self.key_projection(hidden_states[:, start:stop])
                )
                raw_logits = self._raw_feature_logits(key_chunk)
                chunk_max = raw_logits.amax(dim=(-2, -1), keepdim=True)
                key_max = (
                    chunk_max if key_max is None else torch.maximum(key_max, chunk_max)
                )

        assert key_max is not None
        key_max = key_max.detach()

        # 遍历 2
        # 累加 K 和 V 的充分统计量
        # 计算并累加两个核心矩阵 kv_statistics 和 key_statistics
        kv_statistics = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_random_features,
            self.head_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        key_statistics = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_random_features,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        # 遍历 3
        # 计算 Q 的输出
        # 分块计算 Query，并将其与统计量结合生成最终的注意力输出
        for start, stop in self._chunk_ranges(sequence_length):
            chunk = hidden_states[:, start:stop]
            keys = self._split_heads(self.key_projection(chunk))
            values = self._split_heads(self.value_projection(chunk))
            key_features = self._positive_features(keys, stabilizer=key_max)

            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                values_fp32 = values.float()
                chunk_kv = torch.einsum("bhnm,bhnd->bhmd", key_features, values_fp32)
                chunk_keys = key_features.sum(dim=2)
                kv_statistics = kv_statistics + chunk_kv
                key_statistics = key_statistics + chunk_keys

        output_chunks = []
        for start, stop in self._chunk_ranges(sequence_length):
            queries = self._split_heads(
                self.query_projection(hidden_states[:, start:stop])
            )
            query_features = self._positive_features(queries)

            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                numerator = torch.einsum(
                    "bhnm,bhmd->bhnd", query_features, kv_statistics
                )
                denominator = torch.einsum(
                    "bhnm,bhm->bhn", query_features, key_statistics
                )
                denominator = denominator.clamp_min(self.feature_epsilon)
                attended = numerator / denominator.unsqueeze(-1)

            attended = self._merge_heads(attended).to(dtype=output_dtype)
            projected = self.output_projection(attended).to(dtype=output_dtype)
            output_chunks.append(projected)

        return torch.cat(output_chunks, dim=1)


    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, genes, d_model]; "
                f"got rank {hidden_states.ndim}."
            )

        if hidden_states.shape[2] != self.d_model:
            raise ValueError(
                f"Expected hidden width {self.d_model}, got {hidden_states.shape[2]}."
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point.")


    def _chunk_ranges(self, sequence_length: int):
        for start in range(0, sequence_length, self.sequence_chunk_size):
            yield start, min(start + self.sequence_chunk_size, sequence_length)


    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, sequence_length, _ = values.shape
        return values.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)


    def _merge_heads(self, values: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = values.shape
        return values.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)


    def _raw_feature_logits(self, values: Tensor) -> Tensor:
        """Return FP32 ``omega @ x`` terms without retaining a global tensor."""

        with torch.autocast(device_type=values.device.type, enabled=False):
            normalized = values.float() * self._query_key_scale
            projection = self.projection_matrix.float()
            return torch.einsum("bhnd,md->bhnm", normalized, projection)


    def _positive_features(self, values: Tensor, *, stabilizer: Optional[Tensor] = None) -> Tensor:
        """
        执行 FAVOR+ 的核心非线性映射
        Map Q/K to positive FAVOR+ features entirely in FP32.
        Queries use a separate maximum for each token; that multiplicative factor cancels in its numerator/denominator.
        Keys must use the global per-batch/head maximum supplied by the first pass.
        """

        with torch.autocast(device_type=values.device.type, enabled=False):
            normalized = values.float() * self._query_key_scale
            projection = self.projection_matrix.float()
            raw_logits = torch.einsum("bhnd,md->bhnm", normalized, projection)
            squared_norm = normalized.square().sum(dim=-1, keepdim=True) * 0.5
            if stabilizer is None:
                stabilizer = raw_logits.amax(dim=-1, keepdim=True).detach()
            exponent = raw_logits - squared_norm - stabilizer.float()
            return self._feature_scale * (torch.exp(exponent) + self.feature_epsilon)



class PerformerBlock(DenoiserBlock):
    """
    Pre-normalized Performer Block
    The exact residual equations are:
        u = x + dropout(attention(LayerNorm(x)))
        y = u + dropout(ffn(LayerNorm(u)))

    with ffn = Linear(d,4d) -> GELU -> dropout -> Linear(4d,d).
    """

    def __init__(self, config: PerformerConfig, *, layer_index: int) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.layer_index = layer_index
        # ``mixer`` is the stable public name shared with future block types.
        self.mixer_norm = nn.LayerNorm(config.d_model)
        self.mixer = PerformerSelfAttention(config, layer_index=layer_index)
        self.attention_residual_dropout = nn.Dropout(config.dropout)

        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.ffn_residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor, context: DenoiserContext, *, return_diagnostics: bool = False) -> BlockOutput:
        """Output: [B,G,d] and optional detached small diagnostics"""

        self._validate_inputs(hidden_states, context)

        # CUDA autocast may evaluate LayerNorm in FP32.
        # The mixer deliberately preserves the dtype of its direct input, so its branch can therefore return FP32 even when the residual stream is BF16/FP16.
        # Restore the residual-stream dtype before addition: keeping a promoted residual would double long-sequence activation storage and violate the stable block boundary contract.
        # The cast does not change the FP32 feature maps and sufficient statistics computed internally by FAVOR+.
        attention_output = self.mixer(self.mixer_norm(hidden_states)).to(
            dtype=hidden_states.dtype
        )
        residual = hidden_states + self.attention_residual_dropout(attention_output)
        # Apply the same boundary rule independently to the FFN branch because
        # its normalization/nonlinearities are also free to promote under AMP.
        ffn_output = self.ffn(self.ffn_norm(residual)).to(dtype=residual.dtype)
        output = residual + self.ffn_residual_dropout(ffn_output)

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "output_rms": output.detach().float().square().mean().sqrt(),
            }

        return BlockOutput(
            hidden_states=output,
            aux_losses={},
            diagnostics=diagnostics,
        )


    def _validate_inputs(self, hidden_states: Tensor, context: DenoiserContext) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, genes, d_model].")
        batch_size, num_genes, width = hidden_states.shape
        if width != self.d_model:
            raise ValueError(f"Expected hidden width {self.d_model}, got {width}.")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point.")
