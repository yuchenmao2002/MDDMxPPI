"""Contract and numerical tests for the standard Performer backbone.

The tests deliberately use short synthetic sequences.  Their purpose is to
verify the same linear-attention algebra used at 19,295 genes without allocating
large fixtures or making a dense matrix part of the implementation API.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from src.models.backbone import DenoiserBackbone, build_performer_backbone
from src.models.blocks.base import DenoiserBlock
from src.models.blocks.performer import (
    PerformerBlock,
    PerformerSelfAttention,
    make_orthogonal_random_matrix,
)
from src.models.config import PerformerConfig
from src.models.types import BlockOutput, DenoiserContext


def _config(**overrides) -> PerformerConfig:
    values = {
        "num_layers": 2,
        "d_model": 8,
        "head_dim": 4,
        "num_random_features": 32,
        "ffn_multiplier": 2,
        "dropout": 0.0,
        "sequence_chunk_size": 3,
        "projection_seed": 19,
    }
    values.update(overrides)
    return PerformerConfig(**values)


def _context(batch_size: int, num_genes: int, *, masked: bool) -> DenoiserContext:
    return DenoiserContext(
        diffusion_time=torch.linspace(0.0, 1.0, batch_size),
        diffusion_mask=torch.full((batch_size, num_genes), masked, dtype=torch.bool),
    )


def test_random_projection_is_reproducible_without_global_rng_pollution() -> None:
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()

    first = make_orthogonal_random_matrix(11, 4, seed=91)
    rng_after = torch.random.get_rng_state()
    second = make_orthogonal_random_matrix(11, 4, seed=91)
    different = make_orthogonal_random_matrix(11, 4, seed=92)

    assert torch.equal(rng_before, rng_after)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert first.shape == (11, 4)
    assert first.dtype == torch.float32


def test_projection_is_persistent_and_redraw_is_explicit() -> None:
    attention = PerformerSelfAttention(_config(), layer_index=1)
    assert "projection_matrix" in attention.state_dict()
    assert not attention.projection_matrix.requires_grad

    original = attention.projection_matrix.clone()
    rng_before = torch.random.get_rng_state().clone()
    attention.redraw_projection(seed=301)
    redrawn = attention.projection_matrix.clone()
    rng_after = torch.random.get_rng_state()

    assert not torch.equal(original, redrawn)
    assert torch.equal(rng_before, rng_after)
    attention.redraw_projection(seed=301)
    assert torch.equal(attention.projection_matrix, redrawn)


def test_attention_is_chunk_invariant_and_preserves_shape_dtype() -> None:
    chunked = PerformerSelfAttention(_config(sequence_chunk_size=2), layer_index=0)
    unchunked = PerformerSelfAttention(_config(sequence_chunk_size=64), layer_index=0)
    unchunked.load_state_dict(chunked.state_dict())
    chunked.eval()
    unchunked.eval()

    hidden_states = torch.randn(2, 7, 8, dtype=torch.float32)
    chunked_output = chunked(hidden_states)
    unchunked_output = unchunked(hidden_states)

    assert chunked_output.shape == hidden_states.shape
    assert chunked_output.dtype == hidden_states.dtype
    torch.testing.assert_close(chunked_output, unchunked_output, rtol=1e-5, atol=1e-6)


def test_feature_map_and_attention_gradients_remain_finite() -> None:
    attention = PerformerSelfAttention(_config(), layer_index=0)
    feature_input = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    features = attention._positive_features(feature_input)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()

    hidden_states = torch.randn(2, 5, 8, requires_grad=True)
    loss = attention(hidden_states).float().square().mean()
    loss.backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_does_not_call_dense_attention_apis(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("A dense attention API was called.")

    monkeypatch.setattr(torch, "softmax", fail)
    monkeypatch.setattr(torch.nn.functional, "softmax", fail)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail)

    attention = PerformerSelfAttention(_config(), layer_index=0).eval()
    output = attention(torch.randn(1, 9, 8))
    assert output.shape == (1, 9, 8)


def test_positive_features_approximate_small_norm_dense_softmax() -> None:
    # Small-norm inputs keep the Monte Carlo comparison stable while still
    # detecting an incorrect Q/K scaling or denominator contraction.
    config = _config(
        num_layers=1,
        head_dim=8,
        num_random_features=512,
        sequence_chunk_size=2,
    )
    attention = PerformerSelfAttention(config, layer_index=0).eval()
    identity = torch.eye(config.d_model)
    with torch.no_grad():
        attention.query_projection.weight.copy_(identity)
        attention.key_projection.weight.copy_(identity)
        attention.value_projection.weight.copy_(identity)
        attention.output_projection.weight.copy_(identity)
        attention.output_projection.bias.zero_()

    torch.manual_seed(7)
    hidden_states = 0.05 * torch.randn(1, 5, config.d_model)
    approximate = attention(hidden_states)
    scores = torch.einsum("bnd,bmd->bnm", hidden_states, hidden_states) / (
        config.head_dim**0.5
    )
    dense = torch.softmax(scores, dim=-1) @ hidden_states

    torch.testing.assert_close(approximate, dense, rtol=0.20, atol=0.03)


class _ForceFloat32Output(nn.Module):
    """Emulate an AMP branch whose numerically stable operations promote."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.module(hidden_states).float()


def test_block_restores_residual_dtype_after_branch_promotion() -> None:
    """Cover the CUDA LayerNorm promotion failure without requiring a GPU."""

    block = PerformerBlock(_config(num_layers=1), layer_index=0).train()
    block.mixer_norm = _ForceFloat32Output(block.mixer_norm)
    block.ffn = _ForceFloat32Output(block.ffn)
    hidden_states = torch.randn(
        2,
        6,
        block.d_model,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    residual_dtypes = []

    def record_residual_dtype(
        _module: nn.Module,
        inputs: tuple[Tensor, ...],
    ) -> None:
        residual_dtypes.append(inputs[0].dtype)

    hook = block.ffn_norm.register_forward_pre_hook(record_residual_dtype)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = block(hidden_states, _context(2, 6, masked=True))
    finally:
        hook.remove()

    assert residual_dtypes == [hidden_states.dtype]
    assert output.hidden_states.shape == hidden_states.shape
    assert output.hidden_states.device == hidden_states.device
    assert output.hidden_states.dtype == hidden_states.dtype
    assert torch.isfinite(output.hidden_states).all()
    output.hidden_states.float().square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_cuda_bf16_autocast_preserves_block_residual_dtype(
    activation_checkpointing: bool,
) -> None:
    """Exercise the actual CUDA autocast policy that promoted LayerNorm."""

    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    config = _config(
        num_layers=1,
        activation_checkpointing=activation_checkpointing,
        sequence_chunk_size=2,
    )
    backbone = build_performer_backbone(config).cuda().train()
    hidden_states = torch.randn(
        2,
        6,
        config.d_model,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    context = DenoiserContext(
        diffusion_time=torch.tensor([0.25, 0.75], device="cuda"),
        diffusion_mask=torch.zeros(2, 6, dtype=torch.bool, device="cuda"),
    )
    observed_block_dtypes = []

    def record_block_dtype(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        block_output: BlockOutput,
    ) -> None:
        observed_block_dtypes.append(block_output.hidden_states.dtype)

    hook = backbone.blocks[0].register_forward_hook(record_block_dtype)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = backbone(hidden_states, context)
            loss = output.last_hidden_state.float().square().mean()
        loss.backward()
    finally:
        hook.remove()

    assert observed_block_dtypes
    assert set(observed_block_dtypes) == {torch.bfloat16}
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in backbone.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_diffusion_mask_is_context_not_an_attention_padding_mask() -> None:
    block = PerformerBlock(_config(num_layers=1), layer_index=0).eval()
    hidden_states = torch.randn(2, 6, 8)

    visible_output = block(
        hidden_states,
        _context(2, 6, masked=False),
    ).hidden_states
    masked_output = block(
        hidden_states,
        _context(2, 6, masked=True),
    ).hidden_states

    torch.testing.assert_close(visible_output, masked_output, rtol=0.0, atol=0.0)


def test_current_performer_does_not_inject_diffusion_time() -> None:
    block = PerformerBlock(_config(num_layers=1), layer_index=0).eval()
    hidden_states = torch.randn(2, 6, 8)
    diffusion_mask = torch.zeros(2, 6, dtype=torch.bool)
    early = DenoiserContext(
        diffusion_time=torch.tensor([0.1, 0.2], dtype=torch.float32),
        diffusion_mask=diffusion_mask,
    )
    late = DenoiserContext(
        diffusion_time=torch.tensor([0.8, 0.9], dtype=torch.float32),
        diffusion_mask=diffusion_mask,
    )

    early_output = block(hidden_states, early).hidden_states
    late_output = block(hidden_states, late).hidden_states

    torch.testing.assert_close(early_output, late_output, rtol=0.0, atol=0.0)


class _ContractBlock(DenoiserBlock):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = nn.Parameter(torch.tensor(increment, dtype=torch.float32))

    def forward(
        self,
        hidden_states: Tensor,
        context: DenoiserContext,
        *,
        return_diagnostics: bool = False,
    ) -> BlockOutput:
        del context
        output = hidden_states + self.increment
        auxiliary = output.float().mean()
        diagnostics = None
        if return_diagnostics:
            diagnostics = {"mean": output.detach().float().mean()}
        return BlockOutput(
            hidden_states=output,
            aux_losses={"regularizer": auxiliary},
            diagnostics=diagnostics,
        )


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_backbone_namespaces_outputs_and_exposes_normalized_history(
    activation_checkpointing: bool,
) -> None:
    backbone = DenoiserBackbone(
        [_ContractBlock(0.1), _ContractBlock(0.2)],
        d_model=4,
        activation_checkpointing=activation_checkpointing,
    ).train()
    hidden_states = torch.randn(2, 5, 4, requires_grad=True)
    output = backbone(
        hidden_states,
        _context(2, 5, masked=True),
        output_hidden_states=True,
        return_diagnostics=True,
    )

    assert output.last_hidden_state.shape == hidden_states.shape
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 3  # input plus one state per block
    torch.testing.assert_close(output.hidden_states[-1], output.last_hidden_state)
    assert set(output.aux_losses) == {
        "blocks.0/regularizer",
        "blocks.1/regularizer",
    }
    assert output.diagnostics is not None
    assert set(output.diagnostics) == {"blocks.0/mean", "blocks.1/mean"}
    assert all(not value.requires_grad for value in output.diagnostics.values())

    total = output.last_hidden_state.square().mean() + sum(output.aux_losses.values())
    total.backward()
    assert hidden_states.grad is not None
    assert all(block.increment.grad is not None for block in backbone.blocks)


def test_builder_uses_independent_layers_and_stable_state_dict_names() -> None:
    config = _config(num_layers=2)
    backbone = build_performer_backbone(config)

    assert len(backbone.blocks) == 2
    assert backbone.blocks[0] is not backbone.blocks[1]
    assert not torch.equal(
        backbone.blocks[0].mixer.projection_matrix,
        backbone.blocks[1].mixer.projection_matrix,
    )

    keys = set(backbone.state_dict())
    assert "blocks.0.mixer_norm.weight" in keys
    assert "blocks.0.mixer.projection_matrix" in keys
    assert "blocks.0.ffn_norm.weight" in keys
    assert "blocks.0.ffn.0.weight" in keys
    assert "final_norm.weight" in keys


def test_checkpointed_performer_backward_is_finite() -> None:
    config = _config(
        num_layers=1,
        activation_checkpointing=True,
        sequence_chunk_size=2,
    )
    backbone = build_performer_backbone(config).train()
    hidden_states = torch.randn(2, 5, config.d_model, requires_grad=True)

    output = backbone(hidden_states, _context(2, 5, masked=True))
    output.last_hidden_state.float().square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in backbone.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_replacing_chunk_size_does_not_change_config_semantics() -> None:
    config = _config()
    changed = replace(config, sequence_chunk_size=17)
    assert changed.sequence_chunk_size == 17
    assert changed.d_model == config.d_model
