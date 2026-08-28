"""
Gene Identity Encoder
包含由 Geneformer 初始化的源词表，以及从 1152 到模型宽度 d 的共享投影。
不接受每个批次的基因身份索引张量，因为每个样本都使用完全相同的有序轴 [0..19294]。
unbatched 输出必须由顶层模型使用 expand 进行广播，而不是使用 repeat进行复制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from src.models.config import GeneIdentityEncoderConfig
from src.utils.checkpoint import (
    GeneEmbeddingAssetMetadata,
    load_geneformer_embedding_asset,
)



class GeneIdentityEncoder(nn.Module):
    """
    将固定的基因索引映射到 d 维基因身份嵌入
    Parameters
    ----------
    config:
        Dimension, asset and trainability contract.
    initial_weight:
        Float tensor of shape [19295,1152] loaded and validated from the completed Geneformer safetensors asset.
        Implementations must copy this tensor into nn.Embedding and must not rerun the 59-gene donor initialization procedure.
    """

    def __init__(self, config: GeneIdentityEncoderConfig, initial_weight: Tensor) -> None:
        super().__init__()
        self.config = config
        self.asset_metadata: Optional[GeneEmbeddingAssetMetadata] = None

        owned_weight = initial_weight.detach().contiguous().clone()
        self.embedding = nn.Embedding.from_pretrained(
            owned_weight,
            freeze=not config.trainable,  # 是否冻结源词表
        )

        # 降维投影层 [1152,d] / bias = False
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(
                config.source_dim,
                config.d_model,
                bias=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        
        # 确定性正交初始化
        projection_generator = torch.Generator(device="cpu")
        projection_generator.manual_seed(config.projection_seed)
        with torch.no_grad():
            nn.init.orthogonal_(
                self.projection.weight,
                gain=1.0,
                generator=projection_generator,
            )
        
        # 可训练
        self.projection.weight.requires_grad_(True)


    @classmethod
    def from_config(cls, config: GeneIdentityEncoderConfig) -> "GeneIdentityEncoder":
        """加载已审计资产并构建编码器"""

        project_root = Path(__file__).resolve().parents[2]
        initial_weight, metadata = load_geneformer_embedding_asset(
            config.weights_path,
            config.manifest_path,
            tensor_key=config.tensor_key,
            expected_shape=(config.num_genes, config.source_dim),
            verify_sha256=config.verify_sha256,
            project_root=project_root,
        )
        encoder = cls(config, initial_weight)
        encoder.asset_metadata = metadata
        return encoder


    def forward(self) -> Tensor:
        """
        Output:
            基因身份嵌入 [G,d]
        基因索引固定，无需输入
        无批次依赖的前向传播 
        """

        return self.projection(self.embedding.weight)
