"""Audited loading of the two PPI backbone assets.

Both artifacts are produced by ``src/PPI_embedding`` and are loaded, never
regenerated.  Each safetensors file has a same-stem sidecar JSON that records
its schema version, artifact type and ``safetensors_sha256``; this module
validates that record, then validates every tensor key, shape and dtype before
returning CPU tensors.

The two files share one index space: row ``i`` of every tensor is gene ``i`` of
the 19,295-gene project vocabulary, the same axis the denoiser uses for its
``[B,19295,d]`` hidden states.  The spherical asset asserts that alignment
itself through a ``gene_ids`` tensor equal to ``arange(19295)``; the loader
checks it rather than trusting the filename.

Paths are resolved with :func:`src.utils.checkpoint.resolve_project_path`, so
relative asset paths require an explicit project root and no physical
``/home``, ``/nfs/home`` or ``/scratch`` prefix is ever hard-coded here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from src.utils.checkpoint import resolve_project_path, sha256_file


SPHERICAL_SCHEMA_VERSION = "spherical_ppi_embedding.v2"
SPHERICAL_ARTIFACT_TYPE = "spherical_ppi_embedding"
ROUTING_SCHEMA_VERSION = "ppi_moe_static_routing.v1"
ROUTING_ARTIFACT_TYPE = "ppi_moe_static_routing"


@dataclass(frozen=True)
class PPIAssetMetadata:
    """Validated provenance for one loaded pair of PPI artifacts.

    The two SHA-256 strings are the recorded digests of the safetensors files.
    They are provenance only: the tensors themselves travel inside the training
    checkpoint, so inference never needs the original files back.
    """

    embedding_path: Path
    embedding_manifest_path: Path
    embedding_sha256: str
    routing_path: Path
    routing_manifest_path: Path
    routing_sha256: str
    num_genes: int
    embedding_rank: int
    num_experts: int
    route_top_k: int
    embedding_manifest: Dict[str, Any]
    routing_manifest: Dict[str, Any]

    def asset_contract(self) -> Dict[str, Any]:
        """Return the JSON-safe contract stored next to the checkpoint weights."""

        return {
            "embedding_sha256": self.embedding_sha256,
            "routing_sha256": self.routing_sha256,
            "num_genes": self.num_genes,
            "embedding_rank": self.embedding_rank,
            "num_experts": self.num_experts,
            "route_top_k": self.route_top_k,
        }


def _require_regular_file(path: Path, *, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a regular file: {path}")


def _load_sidecar(path: Path, *, description: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} is not valid UTF-8: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{description} root must be a JSON object: {path}")
    return manifest


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(
            f"{description} must be a 64-character hexadecimal string; got {value!r}."
        )
    return value.lower()


def _require_positive_int(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer; got {value!r}.")
    return value


def _check_sidecar_identity(
    manifest: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_artifact: str,
    description: str,
) -> None:
    schema = manifest.get("schema_version")
    if schema != expected_schema:
        raise ValueError(
            f"{description} schema mismatch: expected {expected_schema!r}, "
            f"recorded {schema!r}."
        )
    artifact = manifest.get("artifact_type")
    if artifact != expected_artifact:
        raise ValueError(
            f"{description} artifact type mismatch: expected {expected_artifact!r}, "
            f"recorded {artifact!r}."
        )


def _read_tensors(path: Path, expected_keys: Tuple[str, ...]) -> Dict[str, Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Loading the PPI backbone assets requires the 'safetensors' package."
        ) from exc

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = tuple(handle.keys())
            missing = tuple(key for key in expected_keys if key not in available)
            if missing:
                raise KeyError(
                    f"Tensor keys {missing} are absent from {path}; "
                    f"available keys are {available}."
                )
            return {key: handle.get_tensor(key) for key in expected_keys}
    except KeyError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read PPI safetensors asset {path}: {exc}") from exc


def _check_tensor(
    tensor: Tensor,
    *,
    name: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: torch.dtype,
) -> Tensor:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"PPI tensor {name!r} shape mismatch: expected {expected_shape}, "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"PPI tensor {name!r} must have dtype {expected_dtype}; got {tensor.dtype}."
        )
    if tensor.device.type != "cpu":
        raise ValueError(
            f"PPI tensor {name!r} must be loaded on CPU; got device {tensor.device}."
        )
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"PPI tensor {name!r} contains non-finite values.")
    return tensor.detach().contiguous()


def load_ppi_assets(
    embedding_path: Path,
    embedding_manifest_path: Path,
    routing_path: Path,
    routing_manifest_path: Path,
    *,
    num_genes: int,
    embedding_rank: int,
    num_experts: int,
    route_top_k: int,
    verify_sha256: bool = True,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, Tensor], PPIAssetMetadata]:
    """Return ``(tensors, metadata)`` for the spherical embedding and routing table.

    ``tensors`` contains exactly the five tensors the backbone keeps as buffers:
    ``spherical_embedding``, ``expert_ids``, ``expert_weights``, ``free_mask``
    and ``prototypes``.  Diagnostic-only tensors present in the artifacts
    (``eigenvalues``, ``eigenvectors``, ``row_mean``, ``similarity``) are
    validated for presence where they carry an invariant and otherwise ignored.
    """

    if not isinstance(verify_sha256, bool):
        raise TypeError("verify_sha256 must be a boolean.")
    num_genes = _require_positive_int(num_genes, description="num_genes")
    embedding_rank = _require_positive_int(embedding_rank, description="embedding_rank")
    num_experts = _require_positive_int(num_experts, description="num_experts")
    route_top_k = _require_positive_int(route_top_k, description="route_top_k")
    if route_top_k > num_experts:
        raise ValueError(
            f"route_top_k={route_top_k} cannot exceed num_experts={num_experts}."
        )

    resolved_embedding = resolve_project_path(embedding_path, project_root=project_root)
    resolved_embedding_manifest = resolve_project_path(
        embedding_manifest_path,
        project_root=project_root,
    )
    resolved_routing = resolve_project_path(routing_path, project_root=project_root)
    resolved_routing_manifest = resolve_project_path(
        routing_manifest_path,
        project_root=project_root,
    )
    _require_regular_file(resolved_embedding, description="Spherical PPI embedding")
    _require_regular_file(
        resolved_embedding_manifest,
        description="Spherical PPI embedding sidecar",
    )
    _require_regular_file(resolved_routing, description="PPI MoE routing table")
    _require_regular_file(
        resolved_routing_manifest,
        description="PPI MoE routing sidecar",
    )

    # ---- sidecar records -------------------------------------------------
    embedding_manifest = _load_sidecar(
        resolved_embedding_manifest,
        description="Spherical PPI embedding sidecar",
    )
    _check_sidecar_identity(
        embedding_manifest,
        expected_schema=SPHERICAL_SCHEMA_VERSION,
        expected_artifact=SPHERICAL_ARTIFACT_TYPE,
        description="Spherical PPI embedding sidecar",
    )
    embedding_sha256 = _require_sha256(
        embedding_manifest.get("safetensors_sha256"),
        description="Spherical PPI embedding sidecar safetensors_sha256",
    )
    recorded_genes = embedding_manifest.get("num_genes")
    if recorded_genes != num_genes:
        raise ValueError(
            "Spherical PPI embedding gene count mismatch: "
            f"expected {num_genes}, recorded {recorded_genes!r}."
        )
    recorded_rank = embedding_manifest.get("embedding_rank")
    if recorded_rank != embedding_rank:
        raise ValueError(
            "Spherical PPI embedding rank mismatch: "
            f"expected {embedding_rank}, recorded {recorded_rank!r}."
        )

    routing_manifest = _load_sidecar(
        resolved_routing_manifest,
        description="PPI MoE routing sidecar",
    )
    _check_sidecar_identity(
        routing_manifest,
        expected_schema=ROUTING_SCHEMA_VERSION,
        expected_artifact=ROUTING_ARTIFACT_TYPE,
        description="PPI MoE routing sidecar",
    )
    routing_sha256 = _require_sha256(
        routing_manifest.get("safetensors_sha256"),
        description="PPI MoE routing sidecar safetensors_sha256",
    )
    routing_source = routing_manifest.get("source")
    if not isinstance(routing_source, dict):
        raise ValueError("PPI MoE routing sidecar must contain an object at 'source'.")
    source_sha256 = _require_sha256(
        routing_source.get("embedding_sha256"),
        description="PPI MoE routing sidecar source.embedding_sha256",
    )
    if source_sha256 != embedding_sha256:
        raise ValueError(
            "The routing table was not derived from the selected spherical "
            f"embedding: routing records source {source_sha256}, the embedding "
            f"sidecar records {embedding_sha256}."
        )
    routing_record = routing_manifest.get("routing")
    if not isinstance(routing_record, dict):
        raise ValueError("PPI MoE routing sidecar must contain an object at 'routing'.")
    for field_name, expected in (
        ("num_genes", num_genes),
        ("num_experts", num_experts),
        ("k_route", route_top_k),
    ):
        if routing_record.get(field_name) != expected:
            raise ValueError(
                f"PPI MoE routing {field_name} mismatch: expected {expected}, "
                f"recorded {routing_record.get(field_name)!r}."
            )

    if verify_sha256:
        for resolved, recorded, description in (
            (resolved_embedding, embedding_sha256, "Spherical PPI embedding"),
            (resolved_routing, routing_sha256, "PPI MoE routing table"),
        ):
            actual = sha256_file(resolved)
            if actual != recorded:
                raise ValueError(
                    f"{description} SHA-256 mismatch: expected {recorded}, "
                    f"computed {actual}."
                )

    # ---- tensors ---------------------------------------------------------
    embedding_tensors = _read_tensors(
        resolved_embedding,
        ("embedding", "free_mask", "gene_ids"),
    )
    spherical_embedding = _check_tensor(
        embedding_tensors["embedding"],
        name="embedding",
        expected_shape=(num_genes, embedding_rank),
        expected_dtype=torch.float32,
    )
    embedding_free_mask = _check_tensor(
        embedding_tensors["free_mask"],
        name="free_mask",
        expected_shape=(num_genes,),
        expected_dtype=torch.bool,
    )
    gene_ids = _check_tensor(
        embedding_tensors["gene_ids"],
        name="gene_ids",
        expected_shape=(num_genes,),
        expected_dtype=torch.int64,
    )
    # The artifact asserts its own row alignment; verify it instead of trusting
    # the filename, because every downstream index is taken to be a gene index.
    if not bool(torch.equal(gene_ids, torch.arange(num_genes, dtype=torch.int64))):
        raise ValueError(
            "Spherical PPI embedding gene_ids must equal arange(num_genes); the "
            "asset rows are not aligned with the model gene axis."
        )

    routing_tensors = _read_tensors(
        resolved_routing,
        ("expert_ids", "weights", "free_mask", "prototypes"),
    )
    expert_ids = _check_tensor(
        routing_tensors["expert_ids"],
        name="expert_ids",
        expected_shape=(num_genes, route_top_k),
        expected_dtype=torch.int64,
    )
    expert_weights = _check_tensor(
        routing_tensors["weights"],
        name="weights",
        expected_shape=(num_genes, route_top_k),
        expected_dtype=torch.float32,
    )
    routing_free_mask = _check_tensor(
        routing_tensors["free_mask"],
        name="free_mask",
        expected_shape=(num_genes,),
        expected_dtype=torch.bool,
    )
    prototypes = _check_tensor(
        routing_tensors["prototypes"],
        name="prototypes",
        expected_shape=(num_experts, embedding_rank),
        expected_dtype=torch.float32,
    )

    if not bool(torch.equal(embedding_free_mask, routing_free_mask)):
        raise ValueError(
            "The spherical embedding and the routing table disagree about which "
            "genes carry free (ballast) vectors; they are not a matched pair."
        )
    if int(expert_ids.min()) < 0 or int(expert_ids.max()) >= num_experts:
        raise ValueError(
            f"expert_ids must lie in [0,{num_experts}); got range "
            f"[{int(expert_ids.min())},{int(expert_ids.max())}]."
        )
    # Every gene must be routed to route_top_k distinct experts, otherwise a
    # scatter would double-count one expert's contribution for that gene.
    if int(expert_ids.sort(dim=1).values.diff(dim=1).min()) <= 0:
        raise ValueError("Each row of expert_ids must name distinct experts.")
    if bool((expert_weights < 0).any()):
        raise ValueError("Routing weights must be nonnegative.")
    weight_sums = expert_weights.sum(dim=1)
    if not bool(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)):
        raise ValueError("Each row of the routing weights must sum to one.")

    tensors = {
        "spherical_embedding": spherical_embedding,
        "expert_ids": expert_ids,
        "expert_weights": expert_weights,
        "free_mask": embedding_free_mask,
        "prototypes": prototypes,
    }
    metadata = PPIAssetMetadata(
        embedding_path=resolved_embedding,
        embedding_manifest_path=resolved_embedding_manifest,
        embedding_sha256=embedding_sha256,
        routing_path=resolved_routing,
        routing_manifest_path=resolved_routing_manifest,
        routing_sha256=routing_sha256,
        num_genes=num_genes,
        embedding_rank=embedding_rank,
        num_experts=num_experts,
        route_top_k=route_top_k,
        embedding_manifest=embedding_manifest,
        routing_manifest=routing_manifest,
    )
    return tensors, metadata


__all__ = ["PPIAssetMetadata", "load_ppi_assets"]
