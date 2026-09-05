"""Model-only checkpoint loading contracts for generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

import src.utils.inference_checkpoint as inference_module
from src.models.backbone import build_performer_backbone
from src.models.config import (
    MaskedDiffusionModelConfig,
    PPIAssetConfig,
    PPILFullConfig,
    PerformerConfig,
)
from src.utils.inference_checkpoint import (
    deserialize_model_config,
    load_inference_checkpoint,
)


def _stringify_paths(section: dict, names: tuple) -> None:
    for name in names:
        section[name] = str(section[name])


def _performer_config() -> MaskedDiffusionModelConfig:
    return MaskedDiffusionModelConfig(
        backbone=PerformerConfig(num_layers=1),
        backbone_variant="performer",
    )


def _ppil_config() -> MaskedDiffusionModelConfig:
    return MaskedDiffusionModelConfig(
        backbone=PPILFullConfig(num_layers=1),
        backbone_variant="ppil_full",
        ppi=PPIAssetConfig(),
    )


def _json_config(config: MaskedDiffusionModelConfig = None) -> dict:
    value = asdict(config if config is not None else _performer_config())
    _stringify_paths(value["gene_identity"], ("weights_path", "manifest_path"))
    if value["ppi"] is not None:
        _stringify_paths(
            value["ppi"],
            (
                "embedding_path",
                "embedding_manifest_path",
                "routing_path",
                "routing_manifest_path",
            ),
        )
    return value


def _payload(config: MaskedDiffusionModelConfig = None) -> dict:
    config = config if config is not None else _performer_config()
    contract = None
    if config.ppi is not None:
        contract = {
            "embedding_sha256": "c" * 64,
            "routing_sha256": "d" * 64,
            "num_genes": config.ppi.num_genes,
            "embedding_rank": config.ppi.embedding_rank,
            "num_experts": config.ppi.num_experts,
            "route_top_k": config.ppi.route_top_k,
        }
    return {
        "checkpoint_format_version": 4,
        "architecture_version": config.architecture_version,
        "reason": "epoch_end",
        "model_config": _json_config(config),
        "data_contract": {
            "n_vars": 19_295,
            "gene_order_sha256": "a" * 64,
        },
        "ppi_asset_contract": contract,
        "current_epoch": 2,
        "epoch_completed": True,
        "next_epoch": 3,
        "global_step": 123,
        "primary_validation_metric": "val_time_weighted_hurdle_nll",
        "best_primary_validation_metric": 0.25,
        "model": {"dummy": torch.ones(1)},
        # Deliberately present training-only state; the inference loader must
        # never apply any of it to the fresh model.
        "optimizer": {"sentinel": object()},
        "scheduler": {"sentinel": object()},
        "rng_state": {"sentinel": object()},
    }


class _FakeModel(nn.Module):
    """Stand-in exposing just the ``denoiser.backbone`` the loader inspects."""

    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.denoiser = nn.Module()
        self.denoiser.backbone = build_performer_backbone(
            PerformerConfig(num_layers=num_layers)
        )
        self.extra = nn.Dropout()


def test_deserialize_model_config_restores_nested_types_and_paths() -> None:
    restored = deserialize_model_config(_json_config())

    assert isinstance(restored, MaskedDiffusionModelConfig)
    assert type(restored.backbone) is PerformerConfig
    assert isinstance(restored.gene_identity.weights_path, Path)
    assert restored.backbone.num_layers == 1
    assert restored.backbone_variant == "performer"
    assert restored.ppi is None
    assert restored.architecture_version.startswith("performer*1|")


def test_deserialize_model_config_selects_the_backbone_type_from_the_variant() -> None:
    restored = deserialize_model_config(_json_config(_ppil_config()))

    assert type(restored.backbone) is PPILFullConfig
    assert restored.backbone_variant == "ppil_full"
    assert isinstance(restored.ppi, PPIAssetConfig)
    assert isinstance(restored.ppi.routing_path, Path)
    assert restored.architecture_version.startswith("ppil_full*1|")


def test_deserialize_model_config_rejects_unknown_keys() -> None:
    value = _json_config()
    value["unexpected"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        deserialize_model_config(value)


def test_deserialize_model_config_rejects_the_removed_v3_key_set() -> None:
    value = _json_config()
    value["performer"] = value.pop("backbone")
    value.pop("backbone_variant")
    with pytest.raises(ValueError, match="v4 contract"):
        deserialize_model_config(value)


def test_loader_requires_explicit_checkpoint_trust(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    with pytest.raises(ValueError, match="untrusted"):
        load_inference_checkpoint(path)


def _install_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    *,
    model: nn.Module = None,
    seen: dict = None,
) -> nn.Module:
    fresh_model = model if model is not None else _FakeModel()
    fresh_model.train()

    monkeypatch.setattr(inference_module, "_torch_load_trusted", lambda _p: payload)

    def fake_build(config, state_dict):
        if seen is not None:
            seen["config"] = config
            seen["state_dict"] = state_dict
        return fresh_model

    monkeypatch.setattr(inference_module, "_build_model_from_state", fake_build)
    monkeypatch.setattr(inference_module, "sha256_file", lambda _p: "b" * 64)
    return fresh_model


def test_loader_keeps_only_model_and_audited_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"trusted-test-checkpoint")
    seen: dict = {}
    fresh_model = _install_fake_loader(monkeypatch, _payload(), seen=seen)

    loaded = load_inference_checkpoint(path, trust_checkpoint=True)

    assert loaded.model is fresh_model
    assert not loaded.model.training
    assert seen["state_dict"].keys() == {"dummy"}
    assert loaded.metadata.checkpoint_path == path.resolve()
    assert loaded.metadata.checkpoint_sha256 == "b" * 64
    assert loaded.metadata.current_epoch == 2
    assert loaded.metadata.global_step == 123
    assert loaded.metadata.best_primary_validation_metric == 0.25
    assert loaded.metadata.backbone_variant == "performer"
    assert loaded.metadata.backbone_signature == "performer*1"
    assert loaded.metadata.ppi_asset_contract is None


def test_loader_rejects_gene_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    payload = _payload()
    payload["data_contract"]["n_vars"] = 7
    _install_fake_loader(monkeypatch, payload)

    with pytest.raises(ValueError, match="19295"):
        load_inference_checkpoint(path, trust_checkpoint=True)


def test_loader_rejects_the_superseded_v3_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    payload = _payload()
    payload["checkpoint_format_version"] = 3
    _install_fake_loader(monkeypatch, payload)

    with pytest.raises(ValueError, match="format version"):
        load_inference_checkpoint(path, trust_checkpoint=True)


def test_loader_rejects_an_architecture_from_another_code_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    payload = _payload()
    payload["architecture_version"] = (
        "masked-expression-diffusion-v2-hurdle-truncated-normal"
    )
    _install_fake_loader(monkeypatch, payload)

    with pytest.raises(ValueError, match="three '\\|'-separated segments"):
        load_inference_checkpoint(path, trust_checkpoint=True)


def test_loader_rejects_a_requested_variant_the_checkpoint_does_not_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    _install_fake_loader(monkeypatch, _payload())

    with pytest.raises(ValueError, match="not interchangeable"):
        load_inference_checkpoint(
            path,
            trust_checkpoint=True,
            expected_backbone_variant="ppil_full",
        )


def test_loader_rejects_a_backbone_that_was_not_the_one_built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The identifier is checked against the blocks that really exist.

    The payload claims one Performer layer; the constructed stack has two.  A
    strict state-dict load would also catch this, but only as missing keys.
    """

    path = tmp_path / "checkpoint.pt"
    path.touch()
    _install_fake_loader(monkeypatch, _payload(), model=_FakeModel(num_layers=2))

    with pytest.raises(ValueError, match="Constructed backbone 'performer\\*2'"):
        load_inference_checkpoint(path, trust_checkpoint=True)
