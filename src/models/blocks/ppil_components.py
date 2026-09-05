"""The two PPIL components, implemented once and shared by three block types.

Both read the gene-indexed tables owned by the backbone.  Because those tables
are indexed by gene, and because the model's sequence axis *is* the ordered
gene axis with no padding and no subsetting, a chunk ``[start:stop]`` of hidden
states lines up row-for-row with the same slice of the spherical embedding.
Every component here asserts that the sequence length is the full gene
vocabulary rather than silently trusting it.

Neither component owns the assets.  Each keeps the shared :class:`PPIAssets`
module inside a one-element tuple, which ``nn.Module.__setattr__`` does not
register as a submodule, so stacking six layers does not put six copies of the
tables into the state dict.  The real owner is the backbone, so device
placement still follows ``model.to(...)``.

FP32 discipline follows ``blocks/performer.py``: random-feature maps and all
kernel sufficient statistics are accumulated in FP32 even under BF16 autocast.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from src.models.blocks.base import DenoiserBlock
from src.models.blocks.performer import make_orthogonal_random_matrix
from src.models.ppi_assets import PPIAssets
from src.models.types import BlockOutput, DenoiserContext


class PPILSelfAttention(nn.Module):
    """非因果 PPI 线性注意力

    Same external contract as ``PerformerSelfAttention``: ``[B,G,d] -> [B,G,d]``,
    chunked over the gene axis, with every feature map and kernel sufficient
    statistic accumulated in FP32 even under BF16 autocast.  What differs is the
    kernel, which interpolates content attention with a PPI structural prior::

        q̃_i·k̃_j = (1-λ)·q_i·k_j  +  λ·σ_q·σ_k·(z_i·z_j)

    obtained by feeding augmented vectors to the usual positive feature map::

        q̃_i = [ sqrt(1-λ)·q_i ; sqrt(λ)·σ_q·z_i ]  ∈ R^{d_h+r}
        k̃_j = [ sqrt(1-λ)·k_j ; sqrt(λ)·σ_k·z_j ]

    ``λ(p_t) ∈ (0,1)^h`` is the per-head, per-cell prior share produced by this
    layer's own gate from the realized mask rate.  ``σ_q``, ``σ_k`` are
    stop-gradient within-head RMS norms over the whole gene axis; they calibrate
    the PPI half to the same magnitude as the content half, so that for a fitted
    gene ``E‖q̃_i‖² = σ_q²`` *independently of λ*.  λ therefore reallocates share
    without changing scale, which is why the augmented vector must never be
    rescaled a second time by its own width ``d_h+r``: the only dimensional
    scaling in this class is the single ``d_h^{-1/4}`` applied to the content
    half.

    Free (ballast) genes get a zero vector in the ``r`` block.  Their rows are
    unit vectors in the artifact exactly like the fitted ones, so without this
    they would contribute full-magnitude spurious similarity.  A consequence,
    accepted deliberately: their content half still carries ``sqrt(1-λ)``, so as
    λ grows their logits contract and their attention flattens toward uniform.

    The augmented vector is never materialized.  ``ω·x̃`` splits into a content
    term and a PPI term that is independent of batch and head, so the latter is
    computed once per chunk as ``[n,m]`` and broadcast.
    """

    def __init__(self, config, assets: PPIAssets, *, layer_index: int) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_random_features = config.num_random_features
        self.sequence_chunk_size = config.sequence_chunk_size
        self.feature_epsilon = config.feature_epsilon
        self.layer_index = layer_index

        self.ppi_rank = config.ppi_rank
        self.augmented_head_dim = config.head_dim + config.ppi_rank
        self.num_genes = assets.num_genes
        if assets.embedding_rank != config.ppi_rank:
            raise ValueError(
                f"Spherical embedding has rank {assets.embedding_rank} but the "
                f"block was configured for ppi_rank={config.ppi_rank}."
            )
        # Not registered as a submodule: the backbone owns the single copy.
        self._assets = (assets,)

        # Q/K/V 不使用 biases
        self.query_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output_projection = nn.Linear(config.d_model, config.d_model)

        # 每层独立的先验门控：Fourier(p_t) -> 每个注意力头一个份额
        self.prior_gate = nn.Sequential(
            nn.Linear(2 * config.num_heads, config.gate_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(config.gate_hidden_dim, config.num_heads, bias=True),
        )

        # 投影矩阵作用于增广向量，因此列数是 d_h + r；随机特征数不变。
        projection_seed = config.projection_seed + layer_index
        projection = make_orthogonal_random_matrix(
            config.num_random_features,
            self.augmented_head_dim,
            seed=projection_seed,
        )
        self.register_buffer("projection_matrix", projection, persistent=True)

        self._query_key_scale = config.head_dim**-0.25
        self._feature_scale = config.num_random_features**-0.5

    @property
    def assets(self) -> PPIAssets:
        """The shared, backbone-owned PPI tables."""

        return self._assets[0]

    # ------------------------------------------------------------------
    # PPI prior gate
    # ------------------------------------------------------------------
    def _prior_share(self, mask_rate_features: Tensor):
        """Return ``(λ, sqrt(1-λ), sqrt(λ))`` as FP32 ``[B,H,1,1]`` tensors.

        The two square roots are evaluated as ``exp(½·logsigmoid(∓z))`` rather
        than ``sqrt(sigmoid(∓z))``.  They are mathematically identical, but the
        direct form has an infinite derivative wherever FP32 sigmoid saturates
        to exactly 0 or 1 — reachable for |z| beyond about 17 — which would put
        NaNs into the gradient.  This form saturates smoothly to zero instead,
        and keeps λ strictly inside the open interval the specification states.
        """

        with torch.autocast(device_type=mask_rate_features.device.type, enabled=False):
            logits = self.prior_gate(mask_rate_features.float())  # [B,H]
            share = torch.sigmoid(logits)
            content_scale = torch.exp(0.5 * nn.functional.logsigmoid(-logits))
            ppi_scale = torch.exp(0.5 * nn.functional.logsigmoid(logits))
        shape = (-1, self.num_heads, 1, 1)
        return share.view(shape), content_scale.view(shape), ppi_scale.view(shape)

    # ------------------------------------------------------------------
    # Log-feature coordinates
    # ------------------------------------------------------------------
    def _ppi_projection(self, start: int, stop: int) -> Tensor:
        """FP32 ``ω_ppi @ z_i`` for one gene chunk, shape ``[n,m]``.

        Reads the gated embedding, whose free rows are already zero, and is
        independent of batch and head so it is broadcast into ``[B,H,n,m]``.
        """

        embedding = self.assets.gated_embedding[start:stop].float()
        projection = self.projection_matrix.float()[:, self.head_dim :]
        return torch.einsum("nr,mr->nm", embedding, projection)

    def _log_features(
        self,
        values: Tensor,
        start: int,
        stop: int,
        *,
        sigma: Tensor,
        content_scale: Tensor,
        ppi_scale: Tensor,
    ) -> Tensor:
        """Return FP32 ``a_{·,i,s} = ω_s^T x̃_i − ½‖x̃_i‖²`` for one chunk."""

        with torch.autocast(device_type=values.device.type, enabled=False):
            scaled = values.float() * self._query_key_scale  # the only d_h^{-1/4}
            projection = self.projection_matrix.float()
            content = torch.einsum(
                "bhnd,md->bhnm",
                scaled,
                projection[:, : self.head_dim],
            )
            inner = content_scale * content + (ppi_scale * sigma) * self._ppi_projection(
                start,
                stop,
            )

            # Common unit-sphere template, then the free-gene norm correction.
            # Template and correction together equal the true ‖x̃_i‖²: the
            # template assumes ‖z‖=1, and a free gene's r block is zero.  Adding
            # the correction on top of an already-true norm would inflate a free
            # gene's kernel; there is exactly one norm term here.
            content_square = scaled.square().sum(dim=-1, keepdim=True)
            template = 0.5 * (
                content_scale.square() * content_square + ppi_scale.square() * sigma.square()
            )
            free = self.assets.free_mask_float[start:stop].view(1, 1, -1, 1)
            correction = 0.5 * ppi_scale.square() * sigma.square() * free
            return inner - template + correction

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,
        mask_rate_features: Tensor,
        *,
        diagnostics: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Input & Output: [B,G,d]

        Four chunked passes.  The two RMS statistics are defined over the whole
        gene axis, so they must be complete before any feature map is built:
        ``σ_k`` gates the global key shift ``c_K``, and ``σ_q`` gates the query
        pass.  Pass 0 exists solely to make them exact rather than per-chunk.
        """

        batch_size, sequence_length, _ = hidden_states.shape
        if sequence_length != self.num_genes:
            raise ValueError(
                "PPI linear attention indexes the spherical embedding by gene "
                f"position, so it requires the full {self.num_genes}-gene axis; "
                f"got sequence length {sequence_length}."
            )
        if mask_rate_features is None:
            raise ValueError(
                "PPI linear attention needs the mask-rate features the backbone "
                "derives; they are absent from the denoiser context."
            )
        expected_features = (batch_size, 2 * self.num_heads)
        if tuple(mask_rate_features.shape) != expected_features:
            raise ValueError(
                f"mask_rate_features must have shape {expected_features}, got "
                f"{tuple(mask_rate_features.shape)}."
            )
        output_dtype = hidden_states.dtype

        share, content_scale, ppi_scale = self._prior_share(mask_rate_features)

        # 遍历 0：全序列 RMS 范数。sg[·] 与 no_grad 语义一致。
        with torch.no_grad():
            query_square = torch.zeros(
                batch_size,
                self.num_heads,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            key_square = torch.zeros_like(query_square)
            for start, stop in self._chunk_ranges(sequence_length):
                chunk = hidden_states[:, start:stop]
                queries = self._split_heads(self.query_projection(chunk)).float()
                keys = self._split_heads(self.key_projection(chunk)).float()
                scale_square = self._query_key_scale**2
                query_square += queries.square().sum(dim=(-1, -2)) * scale_square
                key_square += keys.square().sum(dim=(-1, -2)) * scale_square
            sigma_q = (query_square / sequence_length).sqrt().view(-1, self.num_heads, 1, 1)
            sigma_k = (key_square / sequence_length).sqrt().view(-1, self.num_heads, 1, 1)

        # 遍历 1：Key 的全局对数平移。所有 Key 必须共用同一个标量，因为它们
        # 先要汇成一份共享摘要；平移在注意力的分子/分母间约掉。
        key_shift: Optional[Tensor] = None
        with torch.no_grad():
            for start, stop in self._chunk_ranges(sequence_length):
                keys = self._split_heads(
                    self.key_projection(hidden_states[:, start:stop])
                )
                key_logits = self._log_features(
                    keys,
                    start,
                    stop,
                    sigma=sigma_k,
                    content_scale=content_scale,
                    ppi_scale=ppi_scale,
                )
                chunk_max = key_logits.amax(dim=(-2, -1), keepdim=True)
                key_shift = (
                    chunk_max if key_shift is None else torch.maximum(key_shift, chunk_max)
                )

        assert key_shift is not None
        key_shift = key_shift.detach()

        # 遍历 2：累加 Key 与 Value 的充分统计量
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
            key_features = self._positive_features(
                keys,
                start,
                stop,
                sigma=sigma_k,
                content_scale=content_scale,
                ppi_scale=ppi_scale,
                shift=key_shift,
            )

            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                values_fp32 = values.float()
                kv_statistics = kv_statistics + torch.einsum(
                    "bhnm,bhnd->bhmd",
                    key_features,
                    values_fp32,
                )
                key_statistics = key_statistics + key_features.sum(dim=2)

        # 遍历 3：逐块计算 Query 输出。Query 只参与自身输出，因此用逐 token 平移。
        output_chunks = []
        for start, stop in self._chunk_ranges(sequence_length):
            queries = self._split_heads(
                self.query_projection(hidden_states[:, start:stop])
            )
            query_features = self._positive_features(
                queries,
                start,
                stop,
                sigma=sigma_q,
                content_scale=content_scale,
                ppi_scale=ppi_scale,
                shift=None,
            )

            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                numerator = torch.einsum(
                    "bhnm,bhmd->bhnd", query_features, kv_statistics
                )
                denominator = torch.einsum(
                    "bhnm,bhm->bhn", query_features, key_statistics
                )
                # Addition, not clamp_min: it is differentiable everywhere,
                # whereas a clamp has zero gradient below its threshold.
                attended = numerator / (denominator + self.feature_epsilon).unsqueeze(-1)

            attended = self._merge_heads(attended).to(dtype=output_dtype)
            output_chunks.append(self.output_projection(attended).to(dtype=output_dtype))

        if diagnostics is not None:
            flat_share = share.detach().reshape(-1, self.num_heads).float()
            diagnostics["prior_share_mean"] = flat_share.mean()
            diagnostics["prior_share_max"] = flat_share.max()
            diagnostics["sigma_q_mean"] = sigma_q.detach().float().mean()
            diagnostics["sigma_k_mean"] = sigma_k.detach().float().mean()

        return torch.cat(output_chunks, dim=1)

    def _positive_features(
        self,
        values: Tensor,
        start: int,
        stop: int,
        *,
        sigma: Tensor,
        content_scale: Tensor,
        ppi_scale: Tensor,
        shift: Optional[Tensor] = None,
    ) -> Tensor:
        """Stabilized positive features, FP32.

        ``shift=None`` selects the per-token query shift ``c_{Q,i}``; keys pass
        the global ``c_K`` from the first pass.  Both are taken over the
        *complete* log-feature including its norm term, so the largest exponent
        is exactly zero.

        No per-coordinate constant is added to the result: such a constant would
        give the approximated kernel a uniform background unrelated to both
        content and PPI structure, which then accumulates linearly over all
        19,295 keys in the denominator.
        """

        with torch.autocast(device_type=values.device.type, enabled=False):
            logits = self._log_features(
                values,
                start,
                stop,
                sigma=sigma,
                content_scale=content_scale,
                ppi_scale=ppi_scale,
            )
            if shift is None:
                shift = logits.amax(dim=-1, keepdim=True).detach()
            return self._feature_scale * torch.exp(logits - shift.float())

    # ------------------------------------------------------------------
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


class PPIStaticMoEFeedForward(nn.Module):
    """静态路由的专家混合前馈层

    The routing is precomputed offline from the PPI geometry and is fixed for
    the life of the model: gene ``i`` always uses the same ``k`` experts with
    the same convex weights.  There is therefore **no gating network, no
    router logits and no load-balancing auxiliary loss** — the balance was
    already settled when the table was built, so the block returns an empty
    ``aux_losses``.

    Each expert is ``Linear(d, expert_ffn_dim) -> GELU -> Dropout ->
    Linear(expert_ffn_dim, d)``, matching the dense feed-forward network's
    shape.  With the default multiplier of 2 and a top-2 support, every gene
    still sees ``2 * 2d = 4d`` hidden units, so the per-token compute matches
    the dense baseline while total parameters double.

    .. note::

       The expert width is a configuration value, not a decision baked into
       this file; the parameter/compute budget for the ablation is still open.
    """

    def __init__(self, config, assets: PPIAssets, *, layer_index: int) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.route_top_k = config.route_top_k
        self.expert_ffn_dim = config.expert_ffn_dim
        self.layer_index = layer_index
        self.num_genes = assets.num_genes
        # Not registered as a submodule: the backbone owns the single copy.
        self._assets = (assets,)

        if assets.num_experts != self.num_experts:
            raise ValueError(
                f"Routing table provides {assets.num_experts} experts but the "
                f"block was configured for {self.num_experts}."
            )
        if assets.route_top_k != self.route_top_k:
            raise ValueError(
                f"Routing table has top-{assets.route_top_k} support but the "
                f"block was configured for top-{self.route_top_k}."
            )

        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.d_model, self.expert_ffn_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.expert_ffn_dim, config.d_model),
            )
            for _ in range(self.num_experts)
        )

    @property
    def assets(self) -> PPIAssets:
        """The shared, backbone-owned PPI tables."""

        return self._assets[0]

    def forward(
        self,
        hidden_states: Tensor,
        *,
        diagnostics: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Input & Output: [B,G,d].

        ``diagnostics``, when supplied, is filled with one detached FP32 scalar
        per expert rather than changing the return type.
        """

        sequence_length = hidden_states.shape[1]
        if sequence_length != self.num_genes:
            raise ValueError(
                "Static routing is defined per gene, so the feed-forward layer "
                f"requires the full {self.num_genes}-gene axis; got sequence "
                f"length {sequence_length}."
            )

        assets = self.assets
        output = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            gene_index = assets.routing_index(expert_index)
            gene_weight = assets.routing_weight(expert_index)

            selected = hidden_states.index_select(1, gene_index)
            expert_output = expert(selected).to(dtype=hidden_states.dtype)
            weighted = expert_output * gene_weight.to(
                dtype=expert_output.dtype
            ).view(1, -1, 1)
            output.index_add_(1, gene_index, weighted)

            if diagnostics is not None:
                diagnostics[f"expert_{expert_index}_rms"] = (
                    expert_output.detach().float().square().mean().sqrt()
                )

        return output


class PPILBlockBase(DenoiserBlock):
    """Pre-normalized residual shell shared by the three PPIL block variants.

    The exact residual equations are the Performer's::

        u = x + dropout(mixer(LayerNorm(x)))
        y = u + dropout(ffn(LayerNorm(u)))

    so the four backbone variants differ only in which mixer and which
    feed-forward they install.  Subclasses build ``self.mixer`` and ``self.ffn``
    and call :meth:`_install_norms`; everything below is identical across them.

    Static routing produces no auxiliary loss, so ``aux_losses`` stays empty for
    every PPIL variant.
    """

    def _install_norms(self, config, *, mixer_takes_gate: bool, ffn_takes_diagnostics: bool) -> None:
        self.d_model = config.d_model
        self.mixer_norm = nn.LayerNorm(config.d_model)
        self.attention_residual_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn_residual_dropout = nn.Dropout(config.dropout)
        self._mixer_takes_gate = mixer_takes_gate
        self._ffn_takes_diagnostics = ffn_takes_diagnostics

    def forward(
        self,
        hidden_states: Tensor,
        context: DenoiserContext,
        *,
        return_diagnostics: bool = False,
    ) -> BlockOutput:
        """Output: [B,G,d] and optional detached small diagnostics"""

        self._validate_inputs(hidden_states, context)

        diagnostics: Optional[Dict[str, Tensor]] = {} if return_diagnostics else None

        # AMP may promote LayerNorm and the nonlinearities to FP32.  Restore the
        # residual-stream dtype on each branch independently, exactly as the
        # Performer block does: keeping a promoted residual would double
        # long-sequence activation storage and violate the block boundary.
        mixer_input = self.mixer_norm(hidden_states)
        if self._mixer_takes_gate:
            attention_output = self.mixer(
                mixer_input,
                context.mask_rate_features,
                diagnostics=diagnostics,
            )
        else:
            attention_output = self.mixer(mixer_input)
        attention_output = attention_output.to(dtype=hidden_states.dtype)
        residual = hidden_states + self.attention_residual_dropout(attention_output)

        normalized = self.ffn_norm(residual)
        if self._ffn_takes_diagnostics:
            ffn_output = self.ffn(normalized, diagnostics=diagnostics)
        else:
            ffn_output = self.ffn(normalized)
        output = residual + self.ffn_residual_dropout(
            ffn_output.to(dtype=residual.dtype)
        )

        if diagnostics is not None:
            diagnostics["output_rms"] = output.detach().float().square().mean().sqrt()

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
        _, _, width = hidden_states.shape
        if width != self.d_model:
            raise ValueError(f"Expected hidden width {self.d_model}, got {width}.")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point.")


def dense_feed_forward(config) -> nn.Sequential:
    """The baseline position-wise network, identical in shape to the Performer's."""

    return nn.Sequential(
        nn.Linear(config.d_model, config.ffn_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.ffn_dim, config.d_model),
    )


__all__ = [
    "PPILSelfAttention",
    "PPIStaticMoEFeedForward",
    "PPILBlockBase",
    "dense_feed_forward",
]
