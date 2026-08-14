"""Deterministic neural denoiser for masked gene-expression states.

This module owns only the trainable mapping from an explicitly supplied
diffusion state to the parameters of the clean-expression distribution.  It
does not sample forward corruption, evaluate the training objective, or run a
reverse-time sampling loop.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.backbone import DenoiserBackbone, build_performer_backbone
from src.models.config import (
    DEFAULT_D_MODEL,
    NUM_GENES,
    MaskedDiffusionModelConfig,
)
from src.models.gene_expression_decoder import GeneExpressionDecoder
from src.models.gene_expression_encoder import GeneExpressionEncoder
from src.models.gene_identity_encoder import GeneIdentityEncoder
from src.models.masking import AbsorbingStateEmbedding
from src.models.types import DecoderOutput, DenoiserContext, ModelOutput
from src.utils.tensor_validation import (
    validate_diffusion_mask,
    validate_diffusion_time,
    validate_expression_tensor,
    validate_hidden_states,
)


class MaskedExpressionDenoiser(nn.Module):
    """Deterministically predict clean expression from a supplied masked state.

    ``expression_values`` contains the currently visible processed expression
    values.  Values stored at ``diffusion_mask=True`` positions are ignored after
    pointwise encoding and may be arbitrary finite placeholders during future
    sampling.  The boolean mask is the sole source of MASK-state truth.

    Data flow::

        gene_identity = identity_encoder()                 # [G,d]
        expression = expression_encoder(expression_values) # [B,G,d]
        expression_t = absorbing_state(expression, mask)    # [B,G,d]
        hidden = gene_identity[None,...] + expression_t
        hidden = backbone(hidden, context).last_hidden_state
        decoder_output = decoder(hidden)                     # three parameters
        prediction = decoder_output.point_prediction         # optional expectation

    Gene identity broadcast must use a view/``expand``, not physical repetition.
    There is no positional embedding, concatenation or time embedding in v2.
    """

    def __init__(
        self,
        gene_identity_encoder: GeneIdentityEncoder,
        gene_expression_encoder: GeneExpressionEncoder,
        absorbing_state_embedding: AbsorbingStateEmbedding,
        backbone: DenoiserBackbone,
        decoder: GeneExpressionDecoder,
    ) -> None:
        super().__init__()
        self.gene_identity_encoder = gene_identity_encoder
        self.gene_expression_encoder = gene_expression_encoder
        self.absorbing_state_embedding = absorbing_state_embedding
        self.backbone = backbone
        self.decoder = decoder

        self.num_genes = NUM_GENES
        self.d_model = DEFAULT_D_MODEL

    @classmethod
    def from_config(
        cls,
        config: MaskedDiffusionModelConfig,
    ) -> "MaskedExpressionDenoiser":
        """Load initialization assets and assemble the complete v2 denoiser."""

        gene_identity_encoder = GeneIdentityEncoder.from_config(config.gene_identity)
        gene_expression_encoder = GeneExpressionEncoder(config.gene_expression)
        absorbing_state_embedding = AbsorbingStateEmbedding(config.performer.d_model)
        backbone = build_performer_backbone(config.performer)
        decoder = GeneExpressionDecoder(config.decoder)
        return cls(
            gene_identity_encoder=gene_identity_encoder,
            gene_expression_encoder=gene_expression_encoder,
            absorbing_state_embedding=absorbing_state_embedding,
            backbone=backbone,
            decoder=decoder,
        )

    def forward(
        self,
        expression_values: Tensor,
        diffusion_time: Tensor,
        diffusion_mask: Tensor,
        *,
        return_hidden_state: bool = False,
        output_hidden_states: bool = False,
        return_diagnostics: bool = False,
        compute_point_prediction: bool = True,
    ) -> ModelOutput:
        """Predict a clean-expression distribution for a supplied masked state.

        Shapes are ``expression_values[B,19295,1]``, ``diffusion_time[B]`` and
        ``diffusion_mask[B,19295]``.  The v2 Performer does not numerically use time,
        but it remains mandatory in the context and is validated for replayable
        training/sampling semantics.  Direct denoiser calls compute the point
        prediction by default.  NLL-only training explicitly disables it because
        the loss consumes the mandatory hurdle distribution parameters instead.
        """

        if not isinstance(compute_point_prediction, bool):
            raise TypeError("compute_point_prediction must be a boolean.")

        validate_expression_tensor(
            expression_values,
            num_genes=self.num_genes,
            name="expression_values",
            require_nonnegative=False,
        )
        batch_size = expression_values.shape[0]
        validate_diffusion_time(diffusion_time, batch_size=batch_size)
        validate_diffusion_mask(
            diffusion_mask,
            batch_size=batch_size,
            num_genes=self.num_genes,
        )
        if diffusion_time.device != expression_values.device:
            raise ValueError(
                "diffusion_time and expression_values must be on the same device."
            )
        if diffusion_mask.device != expression_values.device:
            raise ValueError(
                "diffusion_mask and expression_values must be on the same device."
            )
        visible_values = expression_values.squeeze(-1).masked_select(~diffusion_mask)
        if (visible_values < 0).any().item():
            raise ValueError(
                "Visible expression values must be nonnegative; values at "
                "diffusion-masked positions may be arbitrary finite placeholders."
            )

        # The pointwise encoder is mathematically discarded at masked positions,
        # but sanitizing first also prevents an arbitrary finite placeholder from
        # overflowing inside a low-precision MLP before torch.where removes it.
        safe_expression_values = expression_values.masked_fill(
            diffusion_mask.unsqueeze(-1),
            0.0,
        )
        expression_embeddings = self.gene_expression_encoder(safe_expression_values)
        validate_hidden_states(
            expression_embeddings,
            num_genes=self.num_genes,
            d_model=self.d_model,
            name="expression_embeddings",
        )
        masked_expression_embeddings = self.absorbing_state_embedding(
            expression_embeddings,
            diffusion_mask,
        )

        gene_embeddings = self.gene_identity_encoder()
        if gene_embeddings.shape != (self.num_genes, self.d_model):
            raise ValueError(
                "gene_identity_encoder must return "
                f"[{self.num_genes},{self.d_model}], got "
                f"{tuple(gene_embeddings.shape)}."
            )
        if gene_embeddings.device != expression_embeddings.device:
            raise ValueError(
                "Gene identity parameters and expression input are on different "
                "devices."
            )
        gene_embeddings = gene_embeddings.to(dtype=expression_embeddings.dtype)
        hidden_states = masked_expression_embeddings + gene_embeddings.unsqueeze(
            0
        ).expand(
            batch_size,
            -1,
            -1,
        )

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
        decoder_output = self.decoder(
            backbone_output.last_hidden_state,
            compute_point_prediction=compute_point_prediction,
        )
        if not isinstance(decoder_output, DecoderOutput):
            raise TypeError(
                "decoder must return DecoderOutput, got "
                f"{type(decoder_output).__name__}."
            )
        point_prediction = decoder_output.point_prediction
        if compute_point_prediction and point_prediction is None:
            raise ValueError(
                "decoder must return point_prediction when it is requested."
            )
        if not compute_point_prediction and point_prediction is not None:
            raise ValueError(
                "decoder must omit point_prediction when it is not requested."
            )
        if point_prediction is not None:
            validate_expression_tensor(
                point_prediction,
                num_genes=self.num_genes,
                name="decoder_output.point_prediction",
                require_nonnegative=True,
            )
            if point_prediction.device != expression_values.device:
                raise ValueError(
                    "Decoder prediction and expression input must share a device."
                )
        if decoder_output.distribution_parameters is None:
            raise ValueError(
                "The hurdle decoder must return distribution_parameters for "
                "training and reverse-process sampling."
            )

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
