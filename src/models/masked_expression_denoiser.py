"""
Denoiser for Masked Gene Expression
This module owns only the trainable mapping from an explicitly supplied diffusion state to the parameters of the clean-expression distribution.
It does not sample forward corruption, evaluate the training objective, or run a reverse-time sampling loop.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from src.models.backbone import DenoiserBackbone, build_denoiser_backbone
from src.models.config import (
    DEFAULT_D_MODEL,
    NUM_GENES,
    MaskedDiffusionModelConfig,
)
from src.models.gene_expression_decoder import GeneExpressionDecoder
from src.models.gene_expression_encoder import GeneExpressionEncoder
from src.models.gene_identity_encoder import GeneIdentityEncoder
from src.models.masking import AbsorbingStateEmbedding
from src.models.ppi_assets import PPIAssets, build_ppi_assets
from src.models.types import DenoiserContext, ModelOutput



class MaskedExpressionDenoiser(nn.Module):
    """
    根据掩码状态确定性地预测干净的基因表达值

    expression_values contains the currently visible processed expression values.
    Values stored at diffusion_mask=True positions are ignored after pointwise encoding and may be arbitrary finite placeholders during future sampling.
    The boolean mask is the sole source of MASK-state truth.
    Data flow::
        gene_identity = identity_encoder()                  # [G,d]
        expression = expression_encoder(expression_values)  # [B,G,d]
        expression_t = absorbing_state(expression, mask)    # [B,G,d]
        hidden = gene_identity[None,...] + expression_t
        hidden = backbone(hidden, context).last_hidden_state
        decoder_output = decoder(hidden)                     # three parameters
        prediction = decoder_output.point_prediction         # optional expectation
    """

    def __init__(self, gene_identity_encoder: GeneIdentityEncoder, gene_expression_encoder: GeneExpressionEncoder, absorbing_state_embedding: AbsorbingStateEmbedding, backbone: DenoiserBackbone, decoder: GeneExpressionDecoder) -> None:
        super().__init__()
        self.gene_identity_encoder = gene_identity_encoder
        self.gene_expression_encoder = gene_expression_encoder
        self.absorbing_state_embedding = absorbing_state_embedding
        self.backbone = backbone
        self.decoder = decoder

        self.num_genes = NUM_GENES
        self.d_model = DEFAULT_D_MODEL


    @classmethod
    def assemble(cls, config: MaskedDiffusionModelConfig, *, gene_identity_encoder: GeneIdentityEncoder, ppi_assets: Optional[PPIAssets]) -> "MaskedExpressionDenoiser":
        """装配去噪器

        The single assembly path.  Training and checkpoint restoration差别只在
        两个外部来源的部件从哪里来——前者读盘，后者从 state dict 取——所以这两个
        部件由调用方注入，其余结构在这里一次性组装。
        Keeping one assembly site is what stops the training-time and
        inference-time models from silently diverging as variants are added.
        """

        return cls(
            gene_identity_encoder=gene_identity_encoder,
            gene_expression_encoder=GeneExpressionEncoder(config.gene_expression),
            absorbing_state_embedding=AbsorbingStateEmbedding(config.backbone.d_model),
            backbone=build_denoiser_backbone(config, ppi_assets=ppi_assets),
            decoder=GeneExpressionDecoder(config.decoder),
        )

    @classmethod
    def from_config(cls, config: MaskedDiffusionModelConfig) -> "MaskedExpressionDenoiser":
        """模型实例化：从磁盘读取并审计外部资产"""

        return cls.assemble(
            config,
            gene_identity_encoder=GeneIdentityEncoder.from_config(config.gene_identity),
            ppi_assets=build_ppi_assets(config, load_from_disk=True),
        )


    def forward(self, expression_values: Tensor, diffusion_time: Tensor, diffusion_mask: Tensor, *, return_hidden_state: bool = False, output_hidden_states: bool = False, return_diagnostics: bool = False, compute_point_prediction: bool = True) -> ModelOutput:
        """
        预测给定掩码状态下的基因表达值分布
        Input:
            expression_values [B,G,1]
            diffusion_time [B]
            diffusion_mask [B,G]
        Output:
        Direct denoiser calls compute the point prediction by default.
        NLL-only training explicitly disables it because the loss consumes the mandatory hurdle distribution parameters instead.
        """

        batch_size = expression_values.shape[0]

        # 数据清洗与安全处理
        safe_expression_values = expression_values.masked_fill(
            diffusion_mask.unsqueeze(-1),
            0.0,
        )

        # 特征编码
        expression_embeddings = self.gene_expression_encoder(safe_expression_values)
        masked_expression_embeddings = self.absorbing_state_embedding(
            expression_embeddings,
            diffusion_mask,
        )
        gene_embeddings = self.gene_identity_encoder()

        gene_embeddings = gene_embeddings.to(dtype=expression_embeddings.dtype)

        # 特征融合
        hidden_states = masked_expression_embeddings + gene_embeddings.unsqueeze(
            0
        ).expand(
            batch_size,
            -1,
            -1,
        )

        # 全局特征提取
        context = DenoiserContext(
            diffusion_time=diffusion_time,
            diffusion_mask=diffusion_mask,
        )
        backbone_output = self.backbone(
            hidden_states,
            context,
            output_hidden_states=output_hidden_states,
            return_diagnostics=return_diagnostics,
        )

        # 解码与输出
        decoder_output = self.decoder(
            backbone_output.last_hidden_state,
            compute_point_prediction=compute_point_prediction,  # 训练阶段仅计算 NLL
        )
        point_prediction = decoder_output.point_prediction

        return ModelOutput(
            prediction=point_prediction,
            last_hidden_state=(
                backbone_output.last_hidden_state if return_hidden_state else None
            ),
            aux_losses=backbone_output.aux_losses,
            diagnostics=backbone_output.diagnostics,
            hidden_states=backbone_output.hidden_states,
            decoder_output=decoder_output,
        )
