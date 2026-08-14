"""Tests for strict, relocation-safe Geneformer asset loading."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.utils.checkpoint import (
    load_geneformer_embedding_asset,
    resolve_project_path,
    sha256_file,
)


def _write_asset(
    root: Path,
    tensor: torch.Tensor,
    *,
    tensor_key: str = "weight",
) -> tuple:
    weights_path = root / "weights.safetensors"
    manifest_path = root / "manifest.json"
    save_file({tensor_key: tensor.contiguous()}, str(weights_path))
    manifest = {
        "schema_version": "1.0",
        "output": {
            "tensor_key": tensor_key,
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype).removeprefix("torch."),
            "file_sha256": sha256_file(weights_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return weights_path, manifest_path


def test_relative_paths_require_and_use_explicit_project_root(tmp_path: Path) -> None:
    nested = tmp_path / "assets"
    nested.mkdir()
    target = nested / "table.bin"
    target.write_bytes(b"asset")

    with pytest.raises(ValueError, match="project_root"):
        resolve_project_path(Path("assets/table.bin"))
    assert (
        resolve_project_path(Path("assets/table.bin"), project_root=tmp_path)
        == target.resolve()
    )
    assert resolve_project_path(target) == target.resolve()


def test_sha256_file_streaming_and_argument_validation(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"abc")
    assert sha256_file(target, chunk_size=1) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(ValueError, match="positive integer"):
        sha256_file(target, chunk_size=0)


def test_loader_validates_and_returns_owned_cpu_float32_tensor(tmp_path: Path) -> None:
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    weights_path, manifest_path = _write_asset(tmp_path, expected)

    loaded, metadata = load_geneformer_embedding_asset(
        weights_path.name,
        manifest_path.name,
        tensor_key="weight",
        expected_shape=(3, 4),
        project_root=tmp_path,
    )

    assert loaded.device.type == "cpu"
    assert loaded.dtype == torch.float32
    assert loaded.is_contiguous()
    assert torch.equal(loaded, expected)
    assert metadata.weights_path == weights_path.resolve()
    assert metadata.manifest_path == manifest_path.resolve()
    assert metadata.shape == (3, 4)
    assert metadata.dtype == "float32"
    assert metadata.file_sha256 == sha256_file(weights_path)


def test_loader_rejects_manifest_and_file_hash_mismatches(tmp_path: Path) -> None:
    weights_path, manifest_path = _write_asset(
        tmp_path, torch.zeros((3, 4), dtype=torch.float32)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["file_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_geneformer_embedding_asset(
            weights_path,
            manifest_path,
            tensor_key="weight",
            expected_shape=(3, 4),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("tensor_key", "other", "tensor key mismatch"),
        ("tensor_shape", [4, 3], "tensor shape mismatch"),
        ("tensor_dtype", "float16", "tensor dtype mismatch"),
    ],
)
def test_loader_rejects_manifest_contract_mismatch(
    tmp_path: Path,
    field: str,
    bad_value,
    match: str,
) -> None:
    weights_path, manifest_path = _write_asset(
        tmp_path, torch.zeros((3, 4), dtype=torch.float32)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"][field] = bad_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_geneformer_embedding_asset(
            weights_path,
            manifest_path,
            tensor_key="weight",
            expected_shape=(3, 4),
            verify_sha256=False,
        )


def test_loader_rejects_nonfinite_tensor(tmp_path: Path) -> None:
    tensor = torch.zeros((3, 4), dtype=torch.float32)
    tensor[0, 0] = float("nan")
    weights_path, manifest_path = _write_asset(tmp_path, tensor)

    with pytest.raises(ValueError, match="non-finite"):
        load_geneformer_embedding_asset(
            weights_path,
            manifest_path,
            tensor_key="weight",
            expected_shape=(3, 4),
        )
