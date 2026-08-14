#!/usr/bin/env python3
"""Initialize the HGNC embedding table from a local Geneformer V2-316M checkpoint."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).absolute()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
PROCESSED_GENEFORMER_DIR = PROJECT_ROOT / "data" / "processed" / "Geneformer"

DEFAULT_MAPPING = PROCESSED_GENEFORMER_DIR / "hgnc_V2_mapping.csv"
DEFAULT_OUTPUT = PROCESSED_GENEFORMER_DIR / "hgnc_V2_embeddings.safetensors"
DEFAULT_MANIFEST = (
    PROCESSED_GENEFORMER_DIR / "hgnc_V2_embeddings_manifest.json"
)

SOURCE_REPO_ID = "ctheodoris/Geneformer"
SOURCE_CHECKPOINT_NAME = "Geneformer-V2-316M"
OFFICIAL_CHECKPOINT_REVISION = "7cf49a35a86ed13714473b420b65bfe4a910de79"
OFFICIAL_CHECKPOINT_SHA256 = (
    "965ceccea81953d362081ef3843560a0e4fef88d396c28017881f1e94b1246f3"
)
DEFAULT_EMBEDDING_KEY = "bert.embeddings.word_embeddings.weight"
DEFAULT_OUTPUT_TENSOR_KEY = "weight"

EXPECTED_SOURCE_SHAPE = (20_275, 1_152)
EXPECTED_OUTPUT_SHAPE = (19_295, 1_152)
EXPECTED_MAPPED_COUNT = 19_236
EXPECTED_MISSING_COUNT = 59
MIN_GENE_TOKEN_ID = 4
DEFAULT_SEED = 42
DEFAULT_NOISE_SCALE = 0.01
DEFAULT_STD_CORRECTION = 0

ENSEMBL_ID_PATTERN = re.compile(r"ENSG\d{11}")
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class MappingEntry:
    symbol: str
    index: int
    ensembl_id: str
    token_id: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Geneformer V2-316M input embeddings according to "
            "hgnc_V2_mapping.csv. Missing genes are initialized from uniformly "
            "sampled mapped rows plus per-dimension Gaussian perturbations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Local checkpoint directory, model.safetensors file, or "
            "model.safetensors.index.json. No network access is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=None,
        help=(
            "Immutable checkpoint revision. It may be omitted only when a commit "
            "can be inferred from a Hugging Face snapshots/<revision> path or "
            "config.json _commit_hash; otherwise the pinned official V2 revision "
            f"{OFFICIAL_CHECKPOINT_REVISION} is used."
        ),
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=OFFICIAL_CHECKPOINT_SHA256,
        help=(
            "Expected SHA-256 for the safetensors shard containing the embedding. "
            f"Default: official V2-316M model hash {OFFICIAL_CHECKPOINT_SHA256}"
        ),
    )
    parser.add_argument(
        "--embedding-key",
        default=DEFAULT_EMBEDDING_KEY,
        help=f"Source embedding tensor key. Default: {DEFAULT_EMBEDDING_KEY}",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help=f"HGNC/V2 mapping CSV. Default: {DEFAULT_MAPPING}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output safetensors file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Output audit manifest JSON. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--output-tensor-key",
        default=DEFAULT_OUTPUT_TENSOR_KEY,
        help=(
            "Tensor key saved in the output file. The default can be loaded "
            f"directly into torch.nn.Embedding. Default: {DEFAULT_OUTPUT_TENSOR_KEY}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Root random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=DEFAULT_NOISE_SCALE,
        help=(
            "Per-dimension perturbation scale applied to empirical sigma. "
            f"Default: {DEFAULT_NOISE_SCALE}"
        ),
    )
    parser.add_argument(
        "--std-correction",
        type=int,
        choices=(0, 1),
        default=DEFAULT_STD_CORRECTION,
        help=(
            "Empirical standard-deviation correction: 0 uses divisor N; "
            "1 uses divisor N-1. Default: 0"
        ),
    )
    return parser.parse_args()


def validate_file(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def validate_checkpoint_path(path: Path) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"Checkpoint is not a file or directory: {path}")
    return path


def comparable_path(path: Path) -> Path:
    return path.expanduser().absolute().resolve(strict=False)


def validate_output_path_collisions(
    output_path: Path,
    manifest_path: Path,
    protected_paths: dict[str, Path | None],
) -> None:
    resolved_output = comparable_path(output_path)
    resolved_manifest = comparable_path(manifest_path)
    if resolved_output == resolved_manifest:
        raise ValueError("--output and --manifest resolve to the same path.")

    for label, protected_path in protected_paths.items():
        if protected_path is None:
            continue
        resolved_protected = comparable_path(protected_path)
        if resolved_output == resolved_protected:
            raise ValueError(f"--output would overwrite {label}: {protected_path}")
        if resolved_manifest == resolved_protected:
            raise ValueError(f"--manifest would overwrite {label}: {protected_path}")


def find_column(fieldnames: list[str], candidates: tuple[str, ...], table_name: str) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    lowered_candidates = {candidate.lower() for candidate in candidates}
    for fieldname in fieldnames:
        if fieldname.lower() in lowered_candidates:
            return fieldname

    joined = ", ".join(candidates)
    raise ValueError(f"{table_name} does not contain any of these columns: {joined}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def load_mapping(path: Path) -> list[MappingEntry]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    symbol_col = find_column(fieldnames, ("Symbol", "symbol"), path.name)
    index_col = find_column(fieldnames, ("Index", "index"), path.name)
    ensembl_col = find_column(
        fieldnames,
        ("Ensembl_ID", "ensembl_id"),
        path.name,
    )
    token_col = find_column(fieldnames, ("Token_ID", "token_id"), path.name)

    entries: list[MappingEntry] = []
    symbols: set[str] = set()
    indices: set[int] = set()
    ensembl_ids: set[str] = set()
    token_ids: set[int] = set()

    for row_number, row in enumerate(rows, start=2):
        symbol = row[symbol_col].strip()
        index_text = row[index_col].strip()
        ensembl_id = row[ensembl_col].strip()
        token_text = row[token_col].strip()

        if not symbol:
            raise ValueError(f"Mapping row {row_number} has an empty Symbol.")
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(
                f"Mapping row {row_number} has a non-integer Index: {index_text!r}"
            ) from error
        if index < 0:
            raise ValueError(f"Mapping row {row_number} has a negative Index: {index}")
        if symbol in symbols:
            raise ValueError(f"Duplicate mapping Symbol: {symbol}")
        if index in indices:
            raise ValueError(f"Duplicate mapping Index: {index}")

        if bool(ensembl_id) != bool(token_text):
            raise ValueError(
                f"Mapping row {row_number} must have both Ensembl_ID and "
                "Token_ID populated or both empty."
            )

        token_id: int | None = None
        if ensembl_id:
            if ENSEMBL_ID_PATTERN.fullmatch(ensembl_id) is None:
                raise ValueError(
                    f"Mapping row {row_number} has an invalid Ensembl_ID: "
                    f"{ensembl_id!r}"
                )
            try:
                token_id = int(token_text)
            except ValueError as error:
                raise ValueError(
                    f"Mapping row {row_number} has a non-integer Token_ID: "
                    f"{token_text!r}"
                ) from error
            if not MIN_GENE_TOKEN_ID <= token_id < EXPECTED_SOURCE_SHAPE[0]:
                raise ValueError(
                    f"Mapping row {row_number} Token_ID {token_id} is outside "
                    f"[{MIN_GENE_TOKEN_ID}, {EXPECTED_SOURCE_SHAPE[0] - 1}]."
                )
            if ensembl_id in ensembl_ids:
                raise ValueError(f"Duplicate mapped Ensembl_ID: {ensembl_id}")
            if token_id in token_ids:
                raise ValueError(f"Duplicate mapped Token_ID: {token_id}")
            ensembl_ids.add(ensembl_id)
            token_ids.add(token_id)

        entries.append(
            MappingEntry(
                symbol=symbol,
                index=index,
                ensembl_id=ensembl_id,
                token_id=token_id,
            )
        )
        symbols.add(symbol)
        indices.add(index)

    if len(entries) != EXPECTED_OUTPUT_SHAPE[0]:
        raise ValueError(
            f"Mapping has {len(entries):,} rows; expected "
            f"{EXPECTED_OUTPUT_SHAPE[0]:,}."
        )
    expected_indices = set(range(EXPECTED_OUTPUT_SHAPE[0]))
    if indices != expected_indices:
        missing = sorted(expected_indices - indices)[:10]
        unexpected = sorted(indices - expected_indices)[:10]
        raise ValueError(
            "Mapping Index values are not exactly 0.."
            f"{EXPECTED_OUTPUT_SHAPE[0] - 1}. Missing examples: {missing}; "
            f"unexpected examples: {unexpected}."
        )

    entries.sort(key=lambda entry: entry.index)
    mapped_count = sum(entry.token_id is not None for entry in entries)
    missing_count = len(entries) - mapped_count
    if mapped_count != EXPECTED_MAPPED_COUNT or missing_count != EXPECTED_MISSING_COUNT:
        raise ValueError(
            f"Mapping contains {mapped_count:,} mapped and {missing_count:,} "
            f"missing rows; expected {EXPECTED_MAPPED_COUNT:,} and "
            f"{EXPECTED_MISSING_COUNT:,}."
        )

    return entries


def import_safetensors():
    try:
        import safetensors
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'safetensors' package is required. Install it in the offline "
            "execution environment before running this script."
        ) from error
    return safetensors.__version__, safe_open, save_file


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def ensure_not_git_lfs_pointer(path: Path) -> None:
    with path.open("rb") as source:
        prefix = source.read(200)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"Checkpoint file is a Git LFS pointer, not model weights: {path}"
        )


def resolve_index_tensor_file(
    index_path: Path,
    embedding_key: str,
) -> Path:
    payload = load_json(index_path, "safetensors index")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Safetensors index has no weight_map object: {index_path}")
    shard_name = weight_map.get(embedding_key)
    if not isinstance(shard_name, str) or not shard_name:
        raise ValueError(
            f"Embedding key {embedding_key!r} is absent from {index_path}."
        )

    index_dir = index_path.parent.resolve()
    tensor_file = (index_path.parent / shard_name).resolve()
    try:
        tensor_file.relative_to(index_dir)
    except ValueError as error:
        raise ValueError(
            f"Safetensors index points outside its directory: {shard_name}"
        ) from error
    return validate_file(tensor_file, "Embedding safetensors shard")


def resolve_embedding_file(
    checkpoint: Path,
    embedding_key: str,
    safe_open,
) -> tuple[Path, Path | None]:
    if checkpoint.is_file():
        if checkpoint.name.endswith(".safetensors.index.json"):
            return resolve_index_tensor_file(checkpoint, embedding_key), checkpoint
        if checkpoint.suffix == ".safetensors":
            return checkpoint, None
        raise ValueError(
            "--checkpoint file must be a .safetensors file or "
            ".safetensors.index.json."
        )

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        return resolve_index_tensor_file(index_path, embedding_key), index_path

    model_path = checkpoint / "model.safetensors"
    if model_path.is_file():
        return model_path, None

    safetensor_files = sorted(checkpoint.glob("*.safetensors"))
    matching_files = []
    for candidate in safetensor_files:
        ensure_not_git_lfs_pointer(candidate)
        with safe_open(str(candidate), framework="np") as source:
            if embedding_key in source.keys():
                matching_files.append(candidate)

    if len(matching_files) == 1:
        return matching_files[0], None
    if not matching_files:
        raise FileNotFoundError(
            f"No safetensors file containing {embedding_key!r} was found in "
            f"{checkpoint}."
        )
    joined = ", ".join(str(path) for path in matching_files)
    raise ValueError(f"Multiple safetensors files contain {embedding_key!r}: {joined}")


def find_config_path(checkpoint: Path, tensor_file: Path) -> Path | None:
    candidates = [tensor_file.parent / "config.json"]
    if checkpoint.is_dir():
        candidates.append(checkpoint / "config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def infer_revision(checkpoint: Path, config: dict[str, Any] | None) -> str | None:
    for candidate in (checkpoint.absolute(), checkpoint.resolve()):
        parts = candidate.parts
        for position, part in enumerate(parts[:-1]):
            if part == "snapshots" and position + 1 < len(parts):
                revision = parts[position + 1]
                if revision:
                    return revision

    if config is not None:
        commit_hash = config.get("_commit_hash")
        if isinstance(commit_hash, str) and commit_hash.strip():
            return commit_hash.strip()
    return None


def normalize_commit_revision(revision: str, source_name: str) -> str:
    normalized = revision.strip().lower()
    if GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{source_name} must be a complete immutable 40-character Git "
            f"commit SHA, got {revision!r}."
        )
    return normalized


def resolve_revision(
    checkpoint: Path,
    declared_revision: str | None,
    config: dict[str, Any] | None,
) -> tuple[str, str | None]:
    declared = (
        normalize_commit_revision(declared_revision, "--checkpoint-revision")
        if declared_revision
        else None
    )
    raw_inferred = infer_revision(checkpoint, config)
    inferred = (
        normalize_commit_revision(raw_inferred, "Inferred checkpoint revision")
        if raw_inferred
        else None
    )
    if declared and inferred and declared != inferred:
        raise ValueError(
            f"Declared checkpoint revision {declared!r} does not match inferred "
            f"revision {inferred!r}."
        )
    revision = declared or inferred or OFFICIAL_CHECKPOINT_REVISION
    return revision, inferred


def validate_config(config: dict[str, Any], config_path: Path) -> None:
    vocab_size = config.get("vocab_size")
    hidden_size = config.get("hidden_size")
    if vocab_size is not None and vocab_size != EXPECTED_SOURCE_SHAPE[0]:
        raise ValueError(
            f"{config_path} vocab_size={vocab_size}; expected "
            f"{EXPECTED_SOURCE_SHAPE[0]}."
        )
    if hidden_size is not None and hidden_size != EXPECTED_SOURCE_SHAPE[1]:
        raise ValueError(
            f"{config_path} hidden_size={hidden_size}; expected "
            f"{EXPECTED_SOURCE_SHAPE[1]}."
        )


def load_source_embedding(
    tensor_file: Path,
    embedding_key: str,
    safe_open,
) -> tuple[np.ndarray, dict[str, str] | None]:
    ensure_not_git_lfs_pointer(tensor_file)
    with safe_open(str(tensor_file), framework="np") as source:
        keys = list(source.keys())
        if embedding_key not in keys:
            embedding_keys = [key for key in keys if "embedding" in key.lower()]
            examples = ", ".join(embedding_keys[:10]) or "(none)"
            raise ValueError(
                f"Embedding key {embedding_key!r} is absent from {tensor_file}. "
                f"Embedding-like keys: {examples}"
            )
        embedding = source.get_tensor(embedding_key)
        metadata = source.metadata()

    if embedding.shape != EXPECTED_SOURCE_SHAPE:
        raise ValueError(
            f"Source embedding shape is {list(embedding.shape)}; expected "
            f"{list(EXPECTED_SOURCE_SHAPE)}."
        )
    if embedding.dtype != np.float32:
        raise ValueError(
            f"Source embedding dtype is {embedding.dtype}; expected float32."
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Source embedding contains NaN or infinite values.")
    return np.array(embedding, dtype=np.float32, copy=True, order="C"), metadata


def initialize_embeddings(
    source_embedding: np.ndarray,
    entries: list[MappingEntry],
    seed: int,
    noise_scale: float,
    std_correction: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[MappingEntry],
    list[MappingEntry],
]:
    if seed < 0:
        raise ValueError("--seed cannot be negative.")
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("--noise-scale must be finite and non-negative.")
    if std_correction not in (0, 1):
        raise ValueError("--std-correction must be 0 or 1.")

    mapped_entries = [entry for entry in entries if entry.token_id is not None]
    missing_entries = [entry for entry in entries if entry.token_id is None]
    if len(mapped_entries) <= std_correction:
        raise ValueError("Not enough mapped rows to calculate empirical standard deviation.")

    mapped_indices = np.asarray(
        [entry.index for entry in mapped_entries],
        dtype=np.int64,
    )
    mapped_token_ids = np.asarray(
        [entry.token_id for entry in mapped_entries],
        dtype=np.int64,
    )
    missing_indices = np.asarray(
        [entry.index for entry in missing_entries],
        dtype=np.int64,
    )

    mapped_vectors = np.ascontiguousarray(source_embedding[mapped_token_ids])
    output = np.empty(
        (len(entries), source_embedding.shape[1]),
        dtype=np.float32,
    )
    output[mapped_indices] = mapped_vectors

    sigma = np.std(
        mapped_vectors,
        axis=0,
        dtype=np.float64,
        ddof=std_correction,
    )
    if not np.isfinite(sigma).all():
        raise ValueError("Empirical per-dimension sigma contains non-finite values.")

    root_sequence = np.random.SeedSequence(seed)
    donor_sequence, noise_sequence = root_sequence.spawn(2)
    donor_rng = np.random.Generator(np.random.PCG64(donor_sequence))
    noise_rng = np.random.Generator(np.random.PCG64(noise_sequence))
    donor_ordinals = donor_rng.integers(
        0,
        len(mapped_entries),
        size=len(missing_entries),
        dtype=np.int64,
    )
    standard_normal = noise_rng.standard_normal(
        size=(len(missing_entries), source_embedding.shape[1])
    )
    noise = standard_normal * (noise_scale * sigma[np.newaxis, :])
    donor_vectors = mapped_vectors[donor_ordinals].astype(np.float64)
    initialized_missing = (donor_vectors + noise).astype(np.float32)
    output[missing_indices] = initialized_missing

    if not np.isfinite(output).all():
        raise ValueError("Initialized output embedding contains non-finite values.")

    return (
        np.ascontiguousarray(output),
        sigma,
        standard_normal,
        noise,
        donor_ordinals,
        mapped_entries,
        missing_entries,
    )


def verify_mapped_rows(
    output: np.ndarray,
    source_embedding: np.ndarray,
    mapped_entries: list[MappingEntry],
    batch_size: int = 2_048,
) -> None:
    for start in range(0, len(mapped_entries), batch_size):
        batch = mapped_entries[start : start + batch_size]
        output_indices = np.asarray([entry.index for entry in batch], dtype=np.int64)
        token_ids = np.asarray([entry.token_id for entry in batch], dtype=np.int64)
        if not np.array_equal(output[output_indices], source_embedding[token_ids]):
            raise ValueError(
                "Mapped output rows do not exactly equal their source Token_ID rows."
            )


def temporary_path(parent: Path, prefix: str, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return Path(name)


def save_and_validate_safetensors(
    output_path: Path,
    tensor_key: str,
    tensor: np.ndarray,
    metadata: dict[str, str],
    safe_open,
    save_file,
) -> tuple[Path, str]:
    temporary_output = temporary_path(
        output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp.safetensors",
    )
    try:
        save_file({tensor_key: np.ascontiguousarray(tensor)}, str(temporary_output), metadata)
        with safe_open(str(temporary_output), framework="np") as source:
            keys = list(source.keys())
            if keys != [tensor_key]:
                raise ValueError(
                    f"Saved safetensors keys are {keys}; expected [{tensor_key!r}]."
                )
            reloaded = source.get_tensor(tensor_key)
        if reloaded.shape != tensor.shape or reloaded.dtype != tensor.dtype:
            raise ValueError("Reloaded output tensor shape or dtype does not match.")
        if not np.array_equal(reloaded, tensor):
            raise ValueError("Reloaded output tensor differs from the in-memory tensor.")
        output_sha256 = sha256_file(temporary_output)
        return temporary_output, output_sha256
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise


def write_manifest_temporary(path: Path, manifest: dict[str, Any]) -> Path:
    temporary_manifest = temporary_path(
        path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.json",
    )
    try:
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(manifest, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        return temporary_manifest
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()

    temporary_output: Path | None = None
    temporary_manifest: Path | None = None
    try:
        checkpoint = validate_checkpoint_path(args.checkpoint)
        mapping_path = validate_file(args.mapping, "Mapping CSV")
        output_path = args.output.expanduser().absolute()
        manifest_path = args.manifest.expanduser().absolute()
        if not args.embedding_key.strip():
            raise ValueError("--embedding-key cannot be empty.")
        if not args.output_tensor_key.strip():
            raise ValueError("--output-tensor-key cannot be empty.")
        if args.expected_checkpoint_sha256 is not None:
            expected_hash = args.expected_checkpoint_sha256.lower()
            if SHA256_PATTERN.fullmatch(expected_hash) is None:
                raise ValueError("--expected-checkpoint-sha256 must be 64 hex digits.")
        else:
            expected_hash = None

        safetensors_version, safe_open, save_file = import_safetensors()
        entries = load_mapping(mapping_path)
        tensor_file, index_path = resolve_embedding_file(
            checkpoint,
            args.embedding_key,
            safe_open,
        )
        ensure_not_git_lfs_pointer(tensor_file)
        config_path = find_config_path(checkpoint, tensor_file)
        config = load_json(config_path, "checkpoint config") if config_path else None
        if config is not None and config_path is not None:
            validate_config(config, config_path)
        validate_output_path_collisions(
            output_path,
            manifest_path,
            {
                "mapping input": mapping_path,
                "checkpoint path": checkpoint,
                "embedding safetensors shard": tensor_file,
                "safetensors index": index_path,
                "checkpoint config": config_path,
                "generator script": SCRIPT_PATH,
            },
        )
        revision, inferred_revision = resolve_revision(
            checkpoint,
            args.checkpoint_revision,
            config,
        )
        if args.checkpoint_revision:
            revision_source = "explicit --checkpoint-revision"
        elif inferred_revision:
            revision_source = "inferred from local checkpoint provenance"
        else:
            revision_source = "pinned official V2-316M revision"

        checkpoint_sha256 = sha256_file(tensor_file)
        if expected_hash is not None and checkpoint_sha256 != expected_hash:
            raise ValueError(
                f"Checkpoint SHA-256 is {checkpoint_sha256}; expected {expected_hash}."
            )
        mapping_sha256 = sha256_file(mapping_path)
        source_embedding, source_metadata = load_source_embedding(
            tensor_file,
            args.embedding_key,
            safe_open,
        )
        source_tensor_sha256 = sha256_array(source_embedding)

        (
            output,
            sigma,
            standard_normal,
            noise,
            donor_ordinals,
            mapped_entries,
            missing_entries,
        ) = initialize_embeddings(
            source_embedding,
            entries,
            seed=args.seed,
            noise_scale=args.noise_scale,
            std_correction=args.std_correction,
        )
        if output.shape != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(
                f"Output shape is {list(output.shape)}; expected "
                f"{list(EXPECTED_OUTPUT_SHAPE)}."
            )
        verify_mapped_rows(output, source_embedding, mapped_entries)
        output_tensor_sha256 = sha256_array(output)

        donor_assignments = []
        for missing_entry, donor_ordinal in zip(missing_entries, donor_ordinals):
            donor = mapped_entries[int(donor_ordinal)]
            donor_assignments.append(
                {
                    "missing_index": missing_entry.index,
                    "missing_symbol": missing_entry.symbol,
                    "donor_ordinal": int(donor_ordinal),
                    "donor_index": donor.index,
                    "donor_symbol": donor.symbol,
                    "donor_ensembl_id": donor.ensembl_id,
                    "donor_token_id": donor.token_id,
                }
            )

        safetensors_metadata = {
            "format": "pt",
            "algorithm": "hgnc_geneformer_v2_embedding_initialization_v1",
            "source_repo_id": SOURCE_REPO_ID,
            "source_checkpoint": SOURCE_CHECKPOINT_NAME,
            "source_revision": revision,
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_embedding_key": args.embedding_key,
            "source_tensor_sha256": source_tensor_sha256,
            "mapping_sha256": mapping_sha256,
            "seed": str(args.seed),
            "noise_scale": repr(args.noise_scale),
            "std_correction": str(args.std_correction),
            "missing_count": str(len(missing_entries)),
            "output_tensor_sha256": output_tensor_sha256,
        }
        temporary_output, output_file_sha256 = save_and_validate_safetensors(
            output_path,
            args.output_tensor_key,
            output,
            safetensors_metadata,
            safe_open,
            save_file,
        )

        config_record = None
        if config_path is not None and config is not None:
            config_record = {
                "path": str(config_path),
                "size_bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
                "vocab_size": config.get("vocab_size"),
                "hidden_size": config.get("hidden_size"),
                "torch_dtype": config.get("torch_dtype"),
                "model_type": config.get("model_type"),
            }

        index_record = None
        if index_path is not None:
            index_record = {
                "path": str(index_path),
                "size_bytes": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
            }

        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "algorithm": {
                "name": "hgnc_geneformer_v2_embedding_initialization",
                "version": "1.0.0",
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": {
                "repo_id": SOURCE_REPO_ID,
                "checkpoint_name": SOURCE_CHECKPOINT_NAME,
                "checkpoint_argument": str(checkpoint),
                "revision": revision,
                "revision_source": revision_source,
                "inferred_revision": inferred_revision,
                "embedding_key": args.embedding_key,
                "tensor_shape": list(source_embedding.shape),
                "tensor_dtype": str(source_embedding.dtype),
                "tensor_sha256": source_tensor_sha256,
                "safetensors_file": {
                    "path": str(tensor_file),
                    "size_bytes": tensor_file.stat().st_size,
                    "sha256": checkpoint_sha256,
                    "expected_sha256": expected_hash,
                    "sha256_verified": expected_hash == checkpoint_sha256,
                },
                "safetensors_index": index_record,
                "safetensors_metadata": source_metadata,
                "config": config_record,
            },
            "mapping": {
                "path": str(mapping_path),
                "size_bytes": mapping_path.stat().st_size,
                "sha256": mapping_sha256,
                "row_count": len(entries),
                "mapped_count": len(mapped_entries),
                "missing_count": len(missing_entries),
                "index_order": "ascending Index; complete permutation 0..19294",
            },
            "initialization": {
                "mapped_rows": "source_embedding[Token_ID]",
                "missing_row_method": "uniform donor sampling with replacement plus Gaussian perturbation",
                "donor_population": "all mapped rows ordered by ascending Index",
                "donor_sampling": "uniform_with_replacement",
                "perturbation_distribution": "independent Normal(0, (noise_scale * sigma_j)^2)",
                "noise_scale": args.noise_scale,
                "sigma_population": "all mapped source vectors",
                "sigma_axis": 0,
                "std_correction": args.std_correction,
                "sigma_compute_dtype": "float64",
                "noise_compute_dtype": "float64",
                "output_dtype": "float32",
                "sigma_sha256": sha256_array(sigma),
                "standard_normal_draws_sha256": sha256_array(standard_normal),
                "noise_sha256": sha256_array(noise),
            },
            "rng": {
                "seed": args.seed,
                "library": "numpy",
                "numpy_version": np.__version__,
                "seed_sequence": "numpy.random.SeedSequence",
                "bit_generator": "PCG64",
                "generator": "numpy.random.Generator",
                "streams": {
                    "donor_sampling_spawn_key": [0],
                    "gaussian_noise_spawn_key": [1],
                },
            },
            "missing_indices": [entry.index for entry in missing_entries],
            "missing_symbols": [entry.symbol for entry in missing_entries],
            "donor_assignments": donor_assignments,
            "output": {
                "path": str(output_path),
                "tensor_key": args.output_tensor_key,
                "tensor_shape": list(output.shape),
                "tensor_dtype": str(output.dtype),
                "tensor_sha256": output_tensor_sha256,
                "file_sha256": output_file_sha256,
                "mapped_rows_exactly_preserved": True,
            },
            "environment": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "byteorder": sys.byteorder,
                "numpy_version": np.__version__,
                "safetensors_version": safetensors_version,
            },
            "script": {
                "path": str(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
                "argv": sys.argv,
            },
        }
        temporary_manifest = write_manifest_temporary(manifest_path, manifest)

        os.replace(temporary_output, output_path)
        temporary_output = None
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None

        print(f"Source embedding: {tensor_file}:{args.embedding_key}")
        print(f"Source shape: {list(EXPECTED_SOURCE_SHAPE)}")
        print(f"Mapped rows copied exactly: {len(mapped_entries):,}")
        print(f"Missing rows initialized: {len(missing_entries):,}")
        print(f"Output tensor: {args.output_tensor_key} {list(output.shape)} {output.dtype}")
        print(f"Output SHA-256: {output_file_sha256}")
        print(f"Output: {output_path}")
        print(f"Manifest: {manifest_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
