"""Shared pointwise encoder for normalized log-expression scalars."""

from __future__ import annotations

from torch import Tensor, nn

from src.models.config import NUM_GENES, GeneExpressionEncoderConfig
from src.utils.tensor_validation import validate_expression_tensor


class GeneExpressionEncoder(nn.Module):
    """Lift each expression scalar independently into model space.

    Architecture: ``Linear(1,32) -> SiLU -> Linear(32,512)``.  Both linear
    layers use bias; there is no normalization, dropout or final activation.
    Parameters are shared across every cell and gene.  This module must never
    mix along the 19,295-token gene axis.
    """

    def __init__(self, config: GeneExpressionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if not config.bias:
            raise ValueError("The expression encoder requires bias=True.")
        self.input_projection = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(
            config.hidden_dim,
            config.d_model,
            bias=True,
        )

    def forward(self, expression_values: Tensor) -> Tensor:
        """Encode ``float[B,19295,1]`` as ``float[B,19295,512]``.

        ``expression_values`` are in the processed
        ``normalize_total(1e4)+log1p`` space.  Numerical zero is a valid clean
        value and receives no special treatment here.
        """

        validate_expression_tensor(
            expression_values,
            num_genes=NUM_GENES,
            # The core denoiser permits arbitrary finite placeholders at MASK
            # positions.  Clean training inputs are checked as non-negative by
            # the training boundary, where the diffusion mask is available.
            require_nonnegative=False,
            expected_device=self.input_projection.weight.device,
        )
        hidden = self.input_projection(expression_values)
        hidden = self.activation(hidden)
        return self.output_projection(hidden)
