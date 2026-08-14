"""Non-causal softmax FAVOR+ Performer block for 19,295 gene tokens.

This module must implement linear attention without ever materializing a dense
``[G,G]`` attention matrix.  Random-feature maps and all kernel sufficient
statistics are accumulated in FP32 even when surrounding projections and
residual activations use BF16 autocast.

The random projection matrix is a persistent, non-trainable buffer owned by
each attention layer.  It is shared across batch items and heads within that
layer.  The current model keeps it fixed for reproducibility; any future redraw
is an explicit trainer action synchronized across distributed ranks, never a
side effect of ``forward``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from src.models.blocks.base import DenoiserBlock
from src.models.config import PerformerConfig
from src.models.types import BlockOutput, DenoiserContext


def make_orthogonal_random_matrix(
    num_rows: int,
    num_columns: int,
    *,
    seed: int,
    device: Optional[object] = None,
) -> Tensor:
    """Construct block-orthogonal Gaussian random features.

    Use QR-derived ``num_columns x num_columns`` blocks, stack and truncate to
    ``num_rows``.  Row lengths follow the chi distribution with
    ``num_columns`` degrees of freedom (the Performer ``ortho_scaling=0``
    convention).  Construction must be deterministic for ``seed`` and must not
    mutate the process-global RNG state.
    """

    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")
    if num_columns <= 0:
        raise ValueError("num_columns must be positive.")

    # Randomness is deliberately generated on CPU with a private generator.
    # Besides preserving the caller's RNG stream, this makes a projection seed
    # independent of the current CUDA device.  Moving the small finished matrix
    # is negligible compared with constructing it during every forward pass.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    num_blocks = math.ceil(num_rows / num_columns)
    unstructured = torch.randn(
        num_blocks,
        num_columns,
        num_columns,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    orthogonal, _ = torch.linalg.qr(unstructured, mode="reduced")
    # Rows, rather than columns, are the random-feature directions.
    matrix = orthogonal.transpose(-2, -1).reshape(-1, num_columns)[:num_rows]
    # A Gaussian vector's length is chi distributed.  Scaling unit orthogonal
    # rows by independently sampled lengths recovers the radial distribution
    # used by the Performer ortho_scaling=0 construction.
    gaussian_rows = torch.randn(
        num_rows,
        num_columns,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    row_lengths = torch.linalg.vector_norm(gaussian_rows, dim=1)
    matrix = matrix * row_lengths.unsqueeze(1)

    if device is not None:
        matrix = matrix.to(device=device)
    return matrix


class PerformerSelfAttention(nn.Module):
    """Bidirectional multi-head softmax attention approximated by FAVOR+.

    Input and output shape: ``[B,G,512]``.  The confirmed configuration uses
    eight heads of width 64 and 256 positive random features.  Q/K are scaled by
    ``head_dim**(-1/4)`` before the positive feature map so the approximated
    kernel is ``exp(q @ k / sqrt(head_dim))``.

    Non-causal attention is evaluated as::

        S = K_prime.transpose(-2,-1) @ V
        z = K_prime.sum(sequence_axis)
        output_i = (Q_prime_i @ S) / (Q_prime_i @ z)

    Sequence chunking bounds the live forward feature-map working set.  Key
    statistics require a global stabilization pass followed by accumulation;
    query outputs are then emitted chunk by chunk.  During training, enabling
    backbone activation checkpointing additionally prevents autograd from
    retaining every chunk's internal features.  No method returns dense
    attention weights.
    """

    def __init__(self, config: PerformerConfig, *, layer_index: int) -> None:
        super().__init__()
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative.")

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_random_features = config.num_random_features
        self.sequence_chunk_size = config.sequence_chunk_size
        self.feature_epsilon = config.feature_epsilon
        self.layer_index = layer_index

        # Separate projections let the global key-stabilization pass avoid
        # computing unused queries and values.  Q/K/V biases are omitted, as in
        # the standard Performer attention parameterization.
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
        self._projection_seed = projection_seed

        self._query_key_scale = config.head_dim**-0.25
        self._feature_scale = config.num_random_features**-0.5

    def redraw_projection(self, seed: int) -> None:
        """Explicitly replace the persistent projection buffer using ``seed``."""

        projection = make_orthogonal_random_matrix(
            self.num_random_features,
            self.head_dim,
            seed=seed,
            device=self.projection_matrix.device,
        )
        # Keep the registered Tensor object (and hence state-dict identity)
        # stable.  Autocast normally leaves this buffer FP32, but respecting its
        # current dtype also makes redraw safe after an explicit module cast.
        self.projection_matrix.copy_(projection.to(self.projection_matrix.dtype))
        self._projection_seed = seed

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply non-causal FAVOR+ self-attention and preserve shape/dtype."""

        self._validate_hidden_states(hidden_states)
        batch_size, sequence_length, _ = hidden_states.shape
        output_dtype = hidden_states.dtype

        # Keys need one common stabilizer per batch/head.  A token-specific key
        # shift would change its weight relative to every other token and would
        # therefore approximate a different kernel.  This detached, no-grad
        # first pass finds a safe global maximum without retaining [B,H,G,m].
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

        # sequence_length is validated as non-zero, so the first pass always
        # initializes key_max.  The assertion also helps static type checkers.
        assert key_max is not None
        key_max = key_max.detach()

        # Accumulate the two non-causal sufficient statistics in FP32.  Their
        # sizes are independent of sequence length: [B,H,m,Dh] and [B,H,m].
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
        if hidden_states.shape[0] <= 0:
            raise ValueError("hidden_states must contain at least one batch item.")
        if hidden_states.shape[1] <= 0:
            raise ValueError("hidden_states must contain at least one gene token.")
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

    def _positive_features(
        self,
        values: Tensor,
        *,
        stabilizer: Optional[Tensor] = None,
    ) -> Tensor:
        """Map Q/K to positive FAVOR+ features entirely in FP32.

        Queries use a separate maximum for each token; that multiplicative
        factor cancels in its numerator/denominator.  Keys must use the global
        per-batch/head maximum supplied by the first pass.
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
    """Pre-normalized Performer attention plus position-wise FFN.

    The exact residual equations are::

        u = x + dropout(attention(LayerNorm(x)))
        y = u + dropout(ffn(LayerNorm(u)))

    with ``ffn = Linear(512,2048) -> GELU -> dropout -> Linear(2048,512)``.
    The current Performer intentionally ignores both context fields, but still
    validates their batch/gene shapes and keeps masked genes active as Q/K/V
    tokens.
    """

    def __init__(self, config: PerformerConfig, *, layer_index: int) -> None:
        super().__init__()
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative.")

        self.d_model = config.d_model
        self.layer_index = layer_index
        # ``mixer`` is the stable public name shared with future block types.
        # Keeping the normalization alongside it also preserves checkpoint keys
        # when a heterogeneous backbone is assembled.
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

    def forward(
        self,
        hidden_states: Tensor,
        context: DenoiserContext,
        *,
        return_diagnostics: bool = False,
    ) -> BlockOutput:
        """Return same-shape hidden states and optional detached small diagnostics."""

        self._validate_inputs(hidden_states, context)

        # CUDA autocast may evaluate LayerNorm in FP32.  The mixer deliberately
        # preserves the dtype of its direct input, so its branch can therefore
        # return FP32 even when the residual stream is BF16/FP16.  Restore the
        # residual-stream dtype before addition: keeping a promoted residual
        # would double long-sequence activation storage and violate the stable
        # block boundary contract.  The cast does not change the FP32 feature
        # maps and sufficient statistics computed internally by FAVOR+.
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
