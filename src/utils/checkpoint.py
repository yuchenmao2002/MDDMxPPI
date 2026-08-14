"""Audited Geneformer initialization loading and checkpoint metadata helpers.

The completed source asset is loaded, never regenerated.  The loader validates
the manifest, tensor key, exact ``[19295,1152]`` shape, float32 dtype and—when
enabled—the recorded SHA-256 before returning a CPU tensor.  Paths are resolved
relative to an explicit project root; code must not hard-code the equivalent
``/home``, ``/nfs/home`` or ``/scratch`` physical path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class GeneEmbeddingAssetMetadata:
    """Validated metadata needed to audit a loaded initialization table."""

    weights_path: Path
    manifest_path: Path
    tensor_key: str
    shape: tuple
    dtype: str
    file_sha256: Optional[str]
    manifest: Dict[str, Any]


def resolve_project_path(path: Path, *, project_root: Optional[Path] = None) -> Path:
    """Resolve an absolute path or a project-root-relative asset path."""

    try:
        candidate = Path(path).expanduser()
    except TypeError as exc:
        raise TypeError(f"path must be path-like, got {type(path).__name__}.") from exc

    if candidate.is_absolute():
        return candidate.resolve()
    if project_root is None:
        raise ValueError(
            f"Relative path {candidate!s} requires an explicit project_root."
        )
    try:
        root = Path(project_root).expanduser().resolve()
    except TypeError as exc:
        raise TypeError(
            f"project_root must be path-like, got {type(project_root).__name__}."
        ) from exc
    return (root / candidate).resolve()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA-256 of a file without loading it all at once."""

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    try:
        file_path = Path(path)
    except TypeError as exc:
        raise TypeError(f"path must be path-like, got {type(path).__name__}.") from exc
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"SHA-256 target is not a regular file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a regular file: {path}")


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in Geneformer manifest {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Geneformer manifest is not valid UTF-8: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Geneformer manifest root must be a JSON object.")
    return manifest


def _normalize_manifest_sha256(value: Any, *, required: bool) -> Optional[str]:
    if value is None and not required:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(
            "Manifest output.file_sha256 must be a 64-character hexadecimal string."
        )
    return value.lower()


def load_geneformer_embedding_asset(
    weights_path: Path,
    manifest_path: Path,
    *,
    tensor_key: str,
    expected_shape: tuple,
    verify_sha256: bool = True,
    project_root: Optional[Path] = None,
) -> tuple:
    """Return ``(weight, metadata)`` after strict asset validation.

    ``weight`` is a contiguous CPU float32 tensor.  The function raises a clear
    exception for missing packages, files, keys or metadata mismatches.  It does
    not alter the asset and does not initialize unmapped genes.
    """

    if not isinstance(tensor_key, str) or not tensor_key:
        raise ValueError("tensor_key must be a non-empty string.")
    if not isinstance(verify_sha256, bool):
        raise TypeError("verify_sha256 must be a boolean.")
    if not isinstance(expected_shape, tuple) or not expected_shape:
        raise TypeError("expected_shape must be a non-empty tuple of dimensions.")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in expected_shape
    ):
        raise ValueError(
            f"expected_shape must contain only positive integers; got {expected_shape!r}."
        )

    resolved_weights = resolve_project_path(weights_path, project_root=project_root)
    resolved_manifest = resolve_project_path(manifest_path, project_root=project_root)
    _require_regular_file(resolved_weights, description="Geneformer weights asset")
    _require_regular_file(resolved_manifest, description="Geneformer manifest")

    manifest = _load_manifest(resolved_manifest)
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("Geneformer manifest must contain an object at 'output'.")

    manifest_key = output.get("tensor_key")
    if manifest_key != tensor_key:
        raise ValueError(
            "Geneformer manifest tensor key mismatch: "
            f"expected {tensor_key!r}, recorded {manifest_key!r}."
        )

    manifest_shape = output.get("tensor_shape")
    if not isinstance(manifest_shape, (list, tuple)) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int)
        for dimension in manifest_shape
    ):
        raise ValueError(
            "Manifest output.tensor_shape must be a list of integer dimensions."
        )
    if tuple(manifest_shape) != expected_shape:
        raise ValueError(
            "Geneformer manifest tensor shape mismatch: "
            f"expected {expected_shape}, recorded {tuple(manifest_shape)}."
        )

    manifest_dtype = output.get("tensor_dtype")
    if manifest_dtype != "float32":
        raise ValueError(
            "Geneformer manifest tensor dtype mismatch: "
            f"expected 'float32', recorded {manifest_dtype!r}."
        )

    recorded_sha256 = _normalize_manifest_sha256(
        output.get("file_sha256"),
        required=verify_sha256,
    )
    if verify_sha256:
        actual_sha256 = sha256_file(resolved_weights)
        if actual_sha256 != recorded_sha256:
            raise ValueError(
                "Geneformer weights SHA-256 mismatch: "
                f"expected {recorded_sha256}, computed {actual_sha256}."
            )

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Loading the Geneformer initialization requires the 'safetensors' package."
        ) from exc

    try:
        with safe_open(
            str(resolved_weights),
            framework="pt",
            device="cpu",
        ) as handle:
            available_keys = tuple(handle.keys())
            if tensor_key not in available_keys:
                raise KeyError(
                    f"Tensor key {tensor_key!r} is absent from {resolved_weights}; "
                    f"available keys are {available_keys}."
                )
            weight = handle.get_tensor(tensor_key)
    except KeyError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to read Geneformer safetensors asset {resolved_weights}: {exc}"
        ) from exc

    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            "Geneformer tensor shape mismatch: "
            f"expected {expected_shape}, got {tuple(weight.shape)}."
        )
    if weight.dtype != torch.float32:
        raise TypeError(
            f"Geneformer tensor must have dtype torch.float32; got {weight.dtype}."
        )
    if weight.device.type != "cpu":
        raise ValueError(
            f"Geneformer tensor must be loaded on CPU; got device {weight.device}."
        )
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("Geneformer tensor contains non-finite values.")

    weight = weight.detach().contiguous()
    metadata = GeneEmbeddingAssetMetadata(
        weights_path=resolved_weights,
        manifest_path=resolved_manifest,
        tensor_key=tensor_key,
        shape=tuple(weight.shape),
        dtype="float32",
        file_sha256=recorded_sha256,
        manifest=manifest,
    )
    return weight, metadata
