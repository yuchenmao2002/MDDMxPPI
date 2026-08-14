"""Model-only checkpoint loading contracts for generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

import src.utils.inference_checkpoint as inference_module
from src.models.config import (
    ARCHITECTURE_VERSION,
    MaskedDiffusionModelConfig,
    PerformerConfig,
)
from src.utils.inference_checkpoint import (
    deserialize_model_config,
    load_inference_checkpoint,
)


def _json_config() -> dict:
    value = asdict(
        MaskedDiffusionModelConfig(performer=PerformerConfig(num_layers=1))
    )
    identity = value["gene_identity"]
    identity["weights_path"] = str(identity["weights_path"])
    identity["manifest_path"] = str(identity["manifest_path"])
    return value


def _payload() -> dict:
    return {
        "checkpoint_format_version": 3,
        "architecture_version": ARCHITECTURE_VERSION,
        "reason": "epoch_end",
        "model_config": _json_config(),
        "data_contract": {
            "n_vars": 19_295,
            "gene_order_sha256": "a" * 64,
        },
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


def test_deserialize_model_config_restores_nested_types_and_paths() -> None:
    restored = deserialize_model_config(_json_config())

    assert isinstance(restored, MaskedDiffusionModelConfig)
    assert isinstance(restored.performer, PerformerConfig)
    assert isinstance(restored.gene_identity.weights_path, Path)
    assert restored.performer.num_layers == 1
    assert restored.architecture_version == ARCHITECTURE_VERSION


def test_deserialize_model_config_rejects_unknown_keys() -> None:
    value = _json_config()
    value["unexpected"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        deserialize_model_config(value)


def test_loader_requires_explicit_checkpoint_trust(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    with pytest.raises(ValueError, match="untrusted"):
        load_inference_checkpoint(path)


def test_loader_keeps_only_model_and_audited_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"trusted-test-checkpoint")
    fresh_model = nn.Sequential(nn.Linear(1, 1), nn.Dropout())
    fresh_model.train()
    seen = {}

    monkeypatch.setattr(inference_module, "_torch_load_trusted", lambda _path: _payload())

    def fake_build(config, state_dict):
        seen["config"] = config
        seen["state_dict"] = state_dict
        return fresh_model

    monkeypatch.setattr(inference_module, "_build_model_from_state", fake_build)
    monkeypatch.setattr(inference_module, "sha256_file", lambda _path: "b" * 64)

    loaded = load_inference_checkpoint(path, trust_checkpoint=True)

    assert loaded.model is fresh_model
    assert not loaded.model.training
    assert seen["state_dict"].keys() == {"dummy"}
    assert loaded.metadata.checkpoint_path == path.resolve()
    assert loaded.metadata.checkpoint_sha256 == "b" * 64
    assert loaded.metadata.current_epoch == 2
    assert loaded.metadata.global_step == 123
    assert loaded.metadata.best_primary_validation_metric == 0.25


def test_loader_rejects_gene_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.touch()
    payload = _payload()
    payload["data_contract"]["n_vars"] = 7
    monkeypatch.setattr(inference_module, "_torch_load_trusted", lambda _path: payload)

    with pytest.raises(ValueError, match="19295"):
        load_inference_checkpoint(path, trust_checkpoint=True)

