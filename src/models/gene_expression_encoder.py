"""Gene Expression Encoder"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.config import NUM_GENES, GeneExpressionEncoderConfig



class GeneExpressionEncoder(nn.Module):
    """
    将每个标量基因表达值独立地升至 d 维基因表达值嵌入
    Architecture: Linear(1,32) -> SiLU -> Linear(32,d). 
    Both linear layers use bias; there is no normalization, dropout or final activation.
    Parameters are shared across every cell and gene. 
    This module must never mix along the 19,295-token gene axis.
    """

    def __init__(self, config: GeneExpressionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # 线性层 [1,32] / bias = True
        self.input_projection = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=True,
        )

        # 激活函数 SiLU
        self.activation = nn.SiLU()

        # 线性层 [32,d] / bias = True
        self.output_projection = nn.Linear(
            config.hidden_dim,
            config.d_model,
            bias=True,
        )


    def forward(self, expression_values: Tensor) -> Tensor:
        """
        Input:
            经 normalize_total(1e4)+log1p 处理的连续基因表达值 [B,G,1]
        Output:
            基因表达值嵌入 [B,G,d]
        Numerical zero is a valid clean value and receives no special treatment here.
        """

        hidden = self.input_projection(expression_values)
        hidden = self.activation(hidden)
        return self.output_projection(hidden)
