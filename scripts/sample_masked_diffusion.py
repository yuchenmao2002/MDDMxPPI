#!/usr/bin/env python3
"""Unconditionally generate processed scRNA expression with one checkpoint.

The command starts every cell at the all-MASK absorbing state and executes a
monotone ``K``-step reverse chain.  All steps use the same frozen checkpoint.
The resulting AnnData ``X`` is in the model's processed PBS expression domain;
it is not an inverse-normalized or integer raw-count matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional


# Match training on networked cluster filesystems. This is set before anndata
# imports h5py so an inherited TRUE value cannot enable problematic locking.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from src.models.config import NUM_GENES
from src.models.reverse_sampler import (
    ReverseSampler,
    SamplingConfig,
    linear_time_grid,
)
from src.utils.checkpoint import sha256_file
from src.utils.inference_checkpoint import (
    InferenceCheckpointMetadata,
    load_inference_checkpoint,
)


DEFAULT_MAPPING = PROJECT_ROOT / "data/processed/PBS/hgnc_pbs_mapping.csv"
GENERATION_SCHEMA_VERSION = "1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate processed scRNA expression from an all-MASK state using "
            "one trained PPIL masked-diffusion checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--schedule",
        choices=("linear",),
        default="linear",
        help="Reverse time grid; v1 supports only the derived linear schedule.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device, normally cuda or cuda:0; cpu is useful for tests.",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--gene-mapping-path",
        type=Path,
        default=DEFAULT_MAPPING,
    )
    parser.add_argument("--gene-index-column", default="Index")
    parser.add_argument("--gene-name-column", default="Symbol")
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help=(
            "Required confirmation that this project checkpoint is trusted. "
            "Training checkpoint v3 requires unrestricted pickle loading."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output file.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_cells", "num_steps", "batch_size"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer.")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("--seed must be a non-negative integer.")
    if args.seed > torch.iinfo(torch.int64).max:
        raise ValueError("--seed must fit in a signed 64-bit integer.")
    if not args.trust_checkpoint:
        raise ValueError(
            "--trust-checkpoint is required. Only load a checkpoint produced by "
            "this project from a trusted location."
        )
    if not args.gene_index_column or not args.gene_name_column:
        raise ValueError("Gene mapping column names must be non-empty.")


def _resolve_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def read_gene_vocabulary(
    path: Path,
    *,
    index_column: str,
    name_column: str,
) -> list[str]:
    """Read the exact ordered 19,295-gene vocabulary used during training."""

    genes: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {index_column, name_column} - fields
        if missing:
            raise ValueError(
                f"Gene mapping is missing columns: {', '.join(sorted(missing))}."
            )
        for expected_index, row in enumerate(reader):
            raw_index = (row.get(index_column) or "").strip()
            gene_name = (row.get(name_column) or "").strip()
            try:
                observed_index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid gene index {raw_index!r} at CSV row "
                    f"{expected_index + 2}."
                ) from exc
            if observed_index != expected_index:
                raise ValueError(
                    f"Gene mapping must be ordered 0..{NUM_GENES - 1}; row "
                    f"{expected_index + 2} contains {observed_index}."
                )
            if not gene_name:
                raise ValueError(f"Empty gene name at CSV row {expected_index + 2}.")
            genes.append(gene_name)
    if len(genes) != NUM_GENES:
        raise ValueError(f"Expected {NUM_GENES} genes, found {len(genes)}.")
    if len(set(genes)) != NUM_GENES:
        raise ValueError("Gene mapping contains duplicate names.")
    return genes


def sha256_strings(values: list[str]) -> str:
    """Match the NUL-delimited gene-order hash stored by the trainer."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid --device {value!r}.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA generation requested, but CUDA is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is not visible.")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only CUDA and CPU generation are currently supported.")
    return device


def autocast_settings(
    *,
    device: torch.device,
    precision: str,
) -> tuple[bool, torch.dtype]:
    if device.type == "cpu":
        if precision != "fp32":
            raise ValueError("CPU generation currently requires --precision fp32.")
        return False, torch.float32
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16.")
        return True, torch.bfloat16
    return False, torch.float32


def generate_expression_matrix(
    *,
    sampler: ReverseSampler,
    num_cells: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    seed: int,
) -> sparse.csr_matrix:
    """Generate in GPU-bounded batches and return exact-zero-preserving CSR."""

    autocast_enabled, autocast_dtype = autocast_settings(
        device=device,
        precision=precision,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batches: list[sparse.csr_matrix] = []
    generated = 0
    start = time.monotonic()
    while generated < num_cells:
        current_batch = min(batch_size, num_cells - generated)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            output = sampler.sample(
                current_batch,
                device=device,
                generator=generator,
                return_diagnostics=False,
                return_trajectory=False,
            )
        values = (
            output.expression_values.squeeze(-1)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        matrix = sparse.csr_matrix(values, dtype=np.float32)
        matrix.eliminate_zeros()
        batches.append(matrix)
        generated += current_batch
        print(
            json.dumps(
                {
                    "event": "sampling_progress",
                    "generated_cells": generated,
                    "num_cells": num_cells,
                    "elapsed_seconds": time.monotonic() - start,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = sparse.vstack(batches, format="csr", dtype=np.float32)
    if result.shape != (num_cells, NUM_GENES):
        raise RuntimeError(
            f"Generated matrix has shape {result.shape}, expected "
            f"({num_cells}, {NUM_GENES})."
        )
    if result.data.size and not np.isfinite(result.data).all():
        raise RuntimeError("Generated matrix contains non-finite values.")
    if result.data.size and np.any(result.data < 0.0):
        raise RuntimeError("Generated matrix contains negative values.")
    return result


def build_anndata(
    matrix: sparse.csr_matrix,
    *,
    genes: list[str],
    metadata: InferenceCheckpointMetadata,
    num_steps: int,
    schedule: str,
    seed: int,
    batch_size: int,
    precision: str,
    device: torch.device,
    mapping_path: Path,
    mapping_sha256: str,
) -> ad.AnnData:
    num_cells = matrix.shape[0]
    obs_names = [f"generated_cell_{index:08d}" for index in range(num_cells)]
    obs = pd.DataFrame(
        {"generated_cell_index": np.arange(num_cells, dtype=np.int64)},
        index=pd.Index(obs_names, name="cell_id"),
    )
    var = pd.DataFrame(
        {"gene_index": np.arange(NUM_GENES, dtype=np.int64)},
        index=pd.Index(genes, name="gene_name"),
    )
    generated = ad.AnnData(X=matrix, obs=obs, var=var)
    generated.uns["ppil_generation"] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "expression_domain": (
            "processed PBS expression domain used to train the checkpoint; "
            "values are not raw integer counts"
        ),
        "checkpoint_path": str(metadata.checkpoint_path),
        "checkpoint_sha256": metadata.checkpoint_sha256,
        "checkpoint_format_version": metadata.checkpoint_format_version,
        "checkpoint_reason": metadata.reason,
        "checkpoint_current_epoch_index": metadata.current_epoch,
        "checkpoint_epoch_completed": metadata.epoch_completed,
        "checkpoint_next_epoch_index": metadata.next_epoch,
        "checkpoint_global_step": metadata.global_step,
        "primary_validation_metric": metadata.primary_validation_metric,
        "best_primary_validation_metric": metadata.best_primary_validation_metric,
        "architecture_version": metadata.architecture_version,
        "num_cells": num_cells,
        "num_steps": num_steps,
        "schedule": schedule,
        "time_grid": linear_time_grid(num_steps).numpy(),
        "seed": seed,
        "batch_size": batch_size,
        "precision": precision,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda or "none",
        "cudnn_version": torch.backends.cudnn.version() or 0,
        "reproducibility_scope": (
            "fixed checkpoint SHA, gene-order SHA, seed, K, batch size, precision, "
            "device class, and software stack; changing batching or runtime may "
            "change the random stream or floating-point result"
        ),
        "gene_mapping_path": str(mapping_path),
        "gene_mapping_sha256": mapping_sha256,
        "gene_order_sha256": sha256_strings(genes),
        "model_config_json": json.dumps(
            _jsonable(asdict(metadata.model_config)), sort_keys=True
        ),
        "training_data_contract_json": json.dumps(
            _jsonable(metadata.data_contract), sort_keys=True
        ),
        "reverse_process": (
            "all-MASK start; independent reveal probability "
            "(t_current-t_next)/t_current; revealed values remain fixed; "
            "unrevealed positions remain MASK and are predicted again"
        ),
    }
    return generated


def atomic_write_h5ad(
    generated: ad.AnnData,
    output_path: Path,
    *,
    compression: Optional[str],
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it."
        )
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}.h5ad"
    )
    try:
        generated.write_h5ad(temporary, compression=compression)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint_path = _resolve_file(args.checkpoint, label="Checkpoint")
    mapping_path = _resolve_file(args.gene_mapping_path, label="Gene mapping")
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".h5ad":
        raise ValueError("--output must have the .h5ad suffix.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"--output exists but is not a regular file: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it."
        )

    genes = read_gene_vocabulary(
        mapping_path,
        index_column=args.gene_index_column,
        name_column=args.gene_name_column,
    )
    gene_hash = sha256_strings(genes)
    mapping_sha256 = sha256_file(mapping_path)
    device = normalize_device(args.device)
    autocast_settings(device=device, precision=args.precision)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    loaded = load_inference_checkpoint(
        checkpoint_path,
        device=device,
        trust_checkpoint=args.trust_checkpoint,
    )
    if not loaded.metadata.epoch_completed:
        raise ValueError(
            "The selected checkpoint was saved during an incomplete epoch. "
            "Use a completed best.pt or completed epoch checkpoint for generation."
        )
    expected_hash = loaded.metadata.data_contract["gene_order_sha256"]
    if gene_hash != expected_hash:
        raise ValueError(
            "Gene mapping order does not match the checkpoint data contract: "
            f"mapping={gene_hash}, checkpoint={expected_hash}."
        )

    sampler = ReverseSampler(
        loaded.model.denoiser,
        SamplingConfig(num_steps=args.num_steps, schedule=args.schedule),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    matrix = generate_expression_matrix(
        sampler=sampler,
        num_cells=args.num_cells,
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
        seed=args.seed,
    )
    generated = build_anndata(
        matrix,
        genes=genes,
        metadata=loaded.metadata,
        num_steps=args.num_steps,
        schedule=args.schedule,
        seed=args.seed,
        batch_size=args.batch_size,
        precision=args.precision,
        device=device,
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha256,
    )
    atomic_write_h5ad(
        generated,
        output_path,
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
    )
    summary = {
        "event": "sampling_complete",
        "output": str(output_path),
        "shape": list(generated.shape),
        "nnz": int(matrix.nnz),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_sha256": loaded.metadata.checkpoint_sha256,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
