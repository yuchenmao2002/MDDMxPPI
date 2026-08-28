#!/usr/bin/env python3
"""Train the masked-expression diffusion model on one CUDA GPU.

The entry point deliberately separates the reusable dataset contract from the
fixed model vocabulary.  It reads an ``.h5ad`` in backed mode, validates the
exact 19,295-gene order against a mapping CSV, and densifies only the rows in a
minibatch.  It is intended for one H100/H200; it does not initialize distributed
training or silently change the model's fixed gene axis.

Checkpoint recovery is exact at completed epoch boundaries.  TERM/INT requests
are handled at the next optimizer-step boundary and save an ``interrupted``
checkpoint.  Resuming such a checkpoint restarts the interrupted epoch with a
deterministic shuffle; already completed updates are retained, so this is safe
recovery rather than a claim of exact mid-epoch replay.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Optional


# Network filesystems used by PBS clusters can hang on HDF5 advisory locking.
# This must be set before importing anndata/h5py in the parent or workers.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
from scipy import sparse
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.models.config import (
    ARCHITECTURE_VERSION,
    NUM_GENES,
    GeneIdentityEncoderConfig,
    MaskedDiffusionModelConfig,
    PerformerConfig,
)
from src.models.masked_diffusion_training import MaskedDiffusionTrainingModule


DEFAULT_DATA = PROJECT_ROOT / "data/processed/PBS/Parse_10M_PBMC_PBS_ln.h5ad"
DEFAULT_MAPPING = PROJECT_ROOT / "data/processed/PBS/hgnc_pbs_mapping.csv"
DEFAULT_GENE_WEIGHTS = (
    PROJECT_ROOT / "data/processed/Geneformer/hgnc_V2_embeddings.safetensors"
)
DEFAULT_GENE_MANIFEST = (
    PROJECT_ROOT / "data/processed/Geneformer/hgnc_V2_embeddings_manifest.json"
)
CHECKPOINT_FORMAT_VERSION = 3
PRIMARY_VALIDATION_METRIC = "val_time_weighted_hurdle_nll"
VALIDATION_SEED_OFFSET = 100_000


class StopRequest:
    """Signal state checked only at safe Python training boundaries."""

    requested = False
    signal_number: Optional[int] = None


STOP_REQUEST = StopRequest()


def _request_stop(signum: int, _frame: object) -> None:
    STOP_REQUEST.requested = True
    STOP_REQUEST.signal_number = signum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the 19,295-gene masked diffusion model on a backed h5ad "
            "using one CUDA GPU."
        )
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    data.add_argument(
        "--matrix-key",
        choices=("X",),
        default="X",
        help=(
            "Backed matrix to read. Only X is accepted because AnnData does not "
            "guarantee backed layer access across workers."
        ),
    )
    data.add_argument(
        "--gene-mapping-path",
        type=Path,
        default=DEFAULT_MAPPING,
        help="CSV containing sequential Index and ordered Symbol columns.",
    )
    data.add_argument("--gene-name-column", default="Symbol")
    data.add_argument("--gene-index-column", default="Index")
    data.add_argument("--val-fraction", type=float, default=0.10)
    data.add_argument(
        "--max-train-cells",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke tests or smaller experiments.",
    )
    data.add_argument("--max-val-cells", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=8)
    data.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    data.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    model = parser.add_argument_group("model")
    model.add_argument("--num-layers", type=int, default=6)
    model.add_argument("--num-random-features", type=int, default=256)
    model.add_argument("--sequence-chunk-size", type=int, default=8192)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    model.add_argument("--gene-weights-path", type=Path, default=DEFAULT_GENE_WEIGHTS)
    model.add_argument(
        "--gene-manifest-path",
        type=Path,
        default=DEFAULT_GENE_MANIFEST,
    )
    model.add_argument("--gene-tensor-key", default="weight")
    model.add_argument(
        "--verify-gene-asset-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument(
        "--torch-compile",
        action="store_true",
        help="Experimental: validation-heavy model contracts may cause graph breaks.",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=10)
    optimization.add_argument("--batch-size", type=int, default=32)
    optimization.add_argument("--grad-accumulation-steps", type=int, default=8)
    optimization.add_argument("--learning-rate", type=float, default=2e-4)
    optimization.add_argument("--weight-decay", type=float, default=0.01)
    optimization.add_argument("--max-grad-norm", type=float, default=1.0)
    optimization.add_argument("--warmup-ratio", type=float, default=0.05)
    optimization.add_argument("--min-lr-ratio", type=float, default=0.1)
    optimization.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    optimization.add_argument("--seed", type=int, default=42)

    runtime = parser.add_argument_group("logging and checkpointing")
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument("--resume", type=Path, default=None)
    runtime.add_argument(
        "--log-every",
        type=int,
        default=50,
        help=(
            "Emit one exact training-window record after this many successful "
            "optimizer updates; a non-empty remainder is emitted at epoch end."
        ),
    )
    runtime.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write an archival epoch checkpoint every N epochs; 0 disables archives.",
    )
    runtime.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Validate every N completed epochs; 0 disables validation.",
    )
    runtime.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help=(
            "Stop after this many consecutive completed validations without a "
            "new primary-metric minimum; 0 disables early stopping."
        ),
    )
    runtime.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=5,
        help="Do not early-stop before this many training epochs are complete.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "epochs",
        "batch_size",
        "grad_accumulation_steps",
        "num_layers",
        "num_random_features",
        "sequence_chunk_size",
        "log_every",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.checkpoint_every < 0 or args.validate_every < 0:
        raise ValueError("Checkpoint and validation intervals must be non-negative.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if args.early_stopping_min_epochs <= 0:
        raise ValueError("--early-stopping-min-epochs must be positive.")
    if args.early_stopping_patience > 0 and args.validate_every == 0:
        raise ValueError(
            "Early stopping requires validation; use --validate-every >= 1 or "
            "disable it with --early-stopping-patience 0."
        )
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must lie strictly between 0 and 1.")
    for name in ("max_train_cells", "max_val_cells"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must lie in [0,1).")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must lie in [0,1].")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must lie in [0,1).")
    if not args.matrix_key:
        raise ValueError("--matrix-key must be non-empty.")


def resolve_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def read_expected_genes(
    path: Path,
    *,
    index_column: str,
    name_column: str,
) -> list[str]:
    """Read and validate the fixed model vocabulary from a mapping CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {index_column, name_column} - fields
        if missing:
            raise ValueError(
                f"Gene mapping is missing columns: {', '.join(sorted(missing))}."
            )
        genes: list[str] = []
        for expected_index, row in enumerate(reader):
            raw_index = (row.get(index_column) or "").strip()
            gene = (row.get(name_column) or "").strip()
            try:
                parsed_index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {index_column}={raw_index!r} at CSV row "
                    f"{expected_index + 2}."
                ) from exc
            if parsed_index != expected_index:
                raise ValueError(
                    f"Gene mapping must be ordered 0..{NUM_GENES - 1}; row "
                    f"{expected_index + 2} contains {parsed_index}."
                )
            if not gene:
                raise ValueError(f"Empty gene name at CSV row {expected_index + 2}.")
            genes.append(gene)

    if len(genes) != NUM_GENES:
        raise ValueError(
            f"Gene mapping must contain {NUM_GENES} genes, got {len(genes)}."
        )
    if len(set(genes)) != NUM_GENES:
        raise ValueError("Gene mapping contains duplicate gene names.")
    return genes


def close_backed(adata: Any) -> None:
    if getattr(adata, "isbacked", False):
        adata.file.close()


def select_matrix(adata: Any, matrix_key: str) -> Any:
    if matrix_key != "X":
        raise ValueError(
            "Only matrix_key='X' is supported for process-safe backed reading. "
            "Write the desired processed layer to X before training."
        )
    return adata.X


class BackedH5adRowDataset(Dataset[Tensor]):
    """Process-safe row dataset that never densifies the complete matrix."""

    def __init__(
        self,
        path: Path,
        matrix_key: str,
        row_indices: np.ndarray,
        expected_genes: list[str],
    ) -> None:
        self.path = path
        self.matrix_key = matrix_key
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self._adata: Any = None
        self._matrix: Any = None
        self._owner_pid: Optional[int] = None

        if self.row_indices.ndim != 1 or self.row_indices.size == 0:
            raise ValueError("Dataset row_indices must be a non-empty vector.")

        adata = ad.read_h5ad(path, backed="r")
        try:
            if adata.n_vars != NUM_GENES:
                raise ValueError(
                    f"Dataset must contain {NUM_GENES} genes, got {adata.n_vars}."
                )
            observed_genes = [str(value) for value in adata.var_names]
            if observed_genes != expected_genes:
                mismatch = next(
                    (
                        position
                        for position, (observed, expected) in enumerate(
                            zip(observed_genes, expected_genes)
                        )
                        if observed != expected
                    ),
                    None,
                )
                if mismatch is None:
                    mismatch_text = "different lengths"
                else:
                    mismatch_text = (
                        f"position {mismatch}: h5ad={observed_genes[mismatch]!r}, "
                        f"mapping={expected_genes[mismatch]!r}"
                    )
                raise ValueError(
                    "H5AD gene order does not match mapping "
                    f"({mismatch_text})."
                )
            matrix = select_matrix(adata, matrix_key)
            if tuple(matrix.shape) != (adata.n_obs, NUM_GENES):
                raise ValueError(
                    f"Selected matrix must have shape [{adata.n_obs},{NUM_GENES}], "
                    f"got {tuple(matrix.shape)}."
                )
            if self.row_indices.min() < 0 or self.row_indices.max() >= adata.n_obs:
                raise IndexError("Dataset row_indices contain an out-of-range cell.")
            self.n_obs_total = int(adata.n_obs)
            self.matrix_dtype = str(matrix.dtype)
        finally:
            close_backed(adata)

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def _ensure_open(self) -> None:
        process_id = os.getpid()
        if self._adata is not None and self._owner_pid == process_id:
            return
        self.close()
        self._adata = ad.read_h5ad(self.path, backed="r")
        self._matrix = select_matrix(self._adata, self.matrix_key)
        self._owner_pid = process_id

    def __getitem__(self, position: int) -> Tensor:
        self._ensure_open()
        source_position = int(self.row_indices[position])
        row = self._matrix[source_position, :]
        if sparse.issparse(row):
            values = row.toarray()
        elif hasattr(row, "toarray"):
            values = row.toarray()
        else:
            values = np.asarray(row)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (NUM_GENES,):
            raise ValueError(
                f"Cell {source_position} produced shape {values.shape}, expected "
                f"({NUM_GENES},)."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Cell {source_position} contains a non-finite value.")
        if np.any(values < 0.0):
            raise ValueError(f"Cell {source_position} contains a negative value.")
        return torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(-1)

    def close(self) -> None:
        if self._adata is not None:
            close_backed(self._adata)
        self._adata = None
        self._matrix = None
        self._owner_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_adata"] = None
        state["_matrix"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deterministic_split(
    n_obs: int,
    val_fraction: float,
    seed: int,
    max_train_cells: Optional[int],
    max_val_cells: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    if n_obs < 2:
        raise ValueError("Training requires at least two cells.")
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_obs)
    val_count = min(n_obs - 1, max(1, int(round(n_obs * val_fraction))))
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]
    if max_train_cells is not None:
        train_indices = train_indices[:max_train_cells]
    if max_val_cells is not None:
        val_indices = val_indices[:max_val_cells]
    # Sorting improves locality; the DataLoader shuffles training positions.
    return np.sort(train_indices), np.sort(val_indices)


def make_loader(
    dataset: BackedH5adRowDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
    persistent_workers: bool,
    sampler_generator: Optional[torch.Generator],
    worker_generator: torch.Generator,
) -> DataLoader[Tensor]:
    options: dict[str, Any] = {}
    if workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=2,
            persistent_workers=persistent_workers,
        )
    sampler = (
        RandomSampler(dataset, generator=sampler_generator) if shuffle else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=worker_generator,
        **options,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_config(args: argparse.Namespace) -> MaskedDiffusionModelConfig:
    performer = PerformerConfig(
        num_layers=args.num_layers,
        num_random_features=args.num_random_features,
        sequence_chunk_size=args.sequence_chunk_size,
        dropout=args.dropout,
        projection_seed=args.seed,
        activation_checkpointing=args.activation_checkpointing,
    )
    identity = GeneIdentityEncoderConfig(
        weights_path=args.gene_weights_path,
        manifest_path=args.gene_manifest_path,
        tensor_key=args.gene_tensor_key,
        projection_seed=args.seed,
        verify_sha256=args.verify_gene_asset_sha256,
    )
    return MaskedDiffusionModelConfig(performer=performer, gene_identity=identity)


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR:
    warmup_steps = int(round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_indices(train_indices: np.ndarray, val_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(train_indices).tobytes())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(val_indices).tobytes())
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng_state(diffusion_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "diffusion_generator": diffusion_generator.get_state(),
    }


def restore_rng_state(
    state: dict[str, Any],
    diffusion_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    diffusion_generator.set_state(state["diffusion_generator"])


def training_signature(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "epochs",
        "batch_size",
        "grad_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
        "min_lr_ratio",
        "precision",
        "seed",
        "val_fraction",
        "max_train_cells",
        "max_val_cells",
        "validate_every",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "torch_compile",
    )
    return {name: jsonable(getattr(args, name)) for name in names}


def checkpoint_payload(
    *,
    model: MaskedDiffusionTrainingModule,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    diffusion_generator: torch.Generator,
    model_config: MaskedDiffusionModelConfig,
    args: argparse.Namespace,
    data_contract: dict[str, Any],
    current_epoch: int,
    epoch_completed: bool,
    next_epoch: int,
    global_step: int,
    best_validation_loss: float,
    early_stopping_bad_validations: int,
    reason: str,
    metrics: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "reason": reason,
        "model_config": jsonable(asdict(model_config)),
        "training_signature": training_signature(args),
        "data_contract": data_contract,
        "current_epoch": current_epoch,
        "epoch_completed": epoch_completed,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "primary_validation_metric": PRIMARY_VALIDATION_METRIC,
        "best_primary_validation_metric": best_validation_loss,
        "early_stopping_bad_validations": early_stopping_bad_validations,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(diffusion_generator),
        "metrics": metrics,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint root must be a dictionary.")
    return payload


def validate_resume_checkpoint(
    payload: dict[str, Any],
    *,
    model_config: MaskedDiffusionModelConfig,
    args: argparse.Namespace,
    data_contract: dict[str, Any],
) -> None:
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version.")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "Checkpoint architecture does not match the hurdle-decoder model."
        )
    if payload.get("model_config") != jsonable(asdict(model_config)):
        raise ValueError("Checkpoint model configuration differs from current CLI.")
    if payload.get("training_signature") != training_signature(args):
        raise ValueError("Checkpoint training schedule differs from current CLI.")
    if payload.get("data_contract") != data_contract:
        raise ValueError("Checkpoint data contract differs from the selected dataset.")
    if payload.get("primary_validation_metric") != PRIMARY_VALIDATION_METRIC:
        raise ValueError("Checkpoint primary validation metric is incompatible.")

    best_metric = payload.get("best_primary_validation_metric")
    if (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or math.isnan(float(best_metric))
    ):
        raise ValueError("Checkpoint best primary validation metric is invalid.")

    current_epoch = payload.get("current_epoch")
    next_epoch = payload.get("next_epoch")
    global_step = payload.get("global_step")
    epoch_completed = payload.get("epoch_completed")
    bad_validations = payload.get("early_stopping_bad_validations")
    integer_fields = {
        "current_epoch": current_epoch,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "early_stopping_bad_validations": bad_validations,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Checkpoint {name} must be an integer.")
    if (
        current_epoch < -1
        or next_epoch < 0
        or global_step < 0
        or bad_validations < 0
    ):
        raise ValueError("Checkpoint epoch/step counters are out of range.")
    if not isinstance(epoch_completed, bool):
        raise ValueError("Checkpoint epoch_completed must be a boolean.")
    expected_next_epoch = current_epoch + 1 if epoch_completed else current_epoch
    if next_epoch != expected_next_epoch:
        raise ValueError(
            "Checkpoint epoch counters disagree with epoch_completed semantics."
        )


_SUFFICIENT_STATISTIC_FIELDS = (
    "weighted_nll_sum",
    "normalizer",
    "weighted_zero_nll_sum",
    "weighted_positive_nll_sum",
    "masked_count",
    "masked_zero_count",
    "masked_positive_count",
    "cell_count",
)


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_nll_sum = 0.0
        self.normalizer = 0
        self.weighted_zero_nll_sum = 0.0
        self.weighted_positive_nll_sum = 0.0
        self.masked_count = 0
        self.masked_zero_count = 0
        self.masked_positive_count = 0
        self.cell_count = 0

    @classmethod
    def from_output(cls, output: Any) -> "MetricAccumulator":
        """Materialize one microbatch's sufficient statistics exactly once."""

        # Stack on the device and transfer once.  Eight separate ``.item()``
        # calls each issue their own synchronizing device-to-host copy in the
        # hot path.  FP64 is exact for both the FP32 sums and the int64 counts,
        # which never approach 2**53.
        packed = torch.stack(
            [
                getattr(output, field).detach().to(dtype=torch.float64)
                for field in _SUFFICIENT_STATISTIC_FIELDS
            ]
        )
        (
            weighted_nll_sum,
            normalizer,
            weighted_zero_nll_sum,
            weighted_positive_nll_sum,
            masked_count,
            masked_zero_count,
            masked_positive_count,
            cell_count,
        ) = packed.tolist()

        accumulator = cls()
        accumulator.weighted_nll_sum = weighted_nll_sum
        accumulator.normalizer = int(normalizer)
        accumulator.weighted_zero_nll_sum = weighted_zero_nll_sum
        accumulator.weighted_positive_nll_sum = weighted_positive_nll_sum
        accumulator.masked_count = int(masked_count)
        accumulator.masked_zero_count = int(masked_zero_count)
        accumulator.masked_positive_count = int(masked_positive_count)
        accumulator.cell_count = int(cell_count)
        return accumulator

    def merge(self, other: "MetricAccumulator") -> None:
        """Add already-materialized sufficient statistics without another sync."""

        if not isinstance(other, MetricAccumulator):
            raise TypeError("other must be a MetricAccumulator.")
        self.weighted_nll_sum += other.weighted_nll_sum
        self.normalizer += other.normalizer
        self.weighted_zero_nll_sum += other.weighted_zero_nll_sum
        self.weighted_positive_nll_sum += other.weighted_positive_nll_sum
        self.masked_count += other.masked_count
        self.masked_zero_count += other.masked_zero_count
        self.masked_positive_count += other.masked_positive_count
        self.cell_count += other.cell_count

    def update(self, output: Any) -> None:
        self.merge(self.from_output(output))

    def result(self) -> dict[str, Any]:
        if self.normalizer <= 0:
            raise RuntimeError("Metric accumulator has a zero normalizer.")
        return {
            "loss": self.weighted_nll_sum / self.normalizer,
            "weighted_nll_sum": self.weighted_nll_sum,
            "normalizer": self.normalizer,
            "weighted_zero_nll_sum": self.weighted_zero_nll_sum,
            "weighted_positive_nll_sum": self.weighted_positive_nll_sum,
            "cell_count": self.cell_count,
            "masked_count": self.masked_count,
            "masked_zero_count": self.masked_zero_count,
            "masked_positive_count": self.masked_positive_count,
        }


def amp_settings(precision: str) -> tuple[bool, torch.dtype]:
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16.")
        return True, torch.bfloat16
    if precision == "fp16":
        return True, torch.float16
    return False, torch.float32


def train_one_epoch(
    *,
    runner: Any,
    model: MaskedDiffusionTrainingModule,
    loader: DataLoader[Tensor],
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    diffusion_generator: torch.Generator,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    accumulation_steps: int,
    max_grad_norm: float,
    log_every: int,
    metrics_path: Path,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, Any], int, bool]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = MetricAccumulator()
    accumulation_group_metrics = MetricAccumulator()
    window_metrics = MetricAccumulator()
    successful_steps_in_window = 0
    attempted_steps_in_window = 0
    clipped_steps_in_window = 0
    last_gradient_norm: Optional[float] = None
    num_batches = len(loader)
    epoch_start = time.monotonic()
    window_start = epoch_start

    def flush_progress_window(*, batch_number: int, reason: str) -> None:
        """Emit one non-overlapping window of sufficient statistics.

        Optimizer attempts skipped by FP16 dynamic loss scaling contribute their
        already-computed loss/cell statistics and elapsed time, but do not advance
        the successful-step interval or the gradient-clipping denominator.
        """

        nonlocal window_metrics
        nonlocal successful_steps_in_window
        nonlocal attempted_steps_in_window
        nonlocal clipped_steps_in_window
        nonlocal last_gradient_norm
        nonlocal window_start

        if window_metrics.normalizer <= 0:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        window_elapsed = time.monotonic() - window_start
        if window_elapsed <= 0.0:
            raise RuntimeError("Training progress window has non-positive time.")
        window = window_metrics.result()
        safe_gradient_norm = (
            last_gradient_norm
            if last_gradient_norm is not None
            and math.isfinite(last_gradient_norm)
            else None
        )
        clip_rate = (
            clipped_steps_in_window / successful_steps_in_window
            if successful_steps_in_window > 0
            else None
        )
        progress = {
            "event": "train_window",
            "epoch": epoch + 1,
            "batch": batch_number,
            "batches": num_batches,
            "global_step": global_step,
            "flush_reason": reason,
            # This is a ratio of summed sufficient statistics, never an
            # arithmetic average of microbatch losses.
            "train_window_loss": window["loss"],
            "loss": window["loss"],
            "window_weighted_nll_sum": window["weighted_nll_sum"],
            "window_normalizer": window["normalizer"],
            "window_cell_count": window["cell_count"],
            "window_optimizer_steps": successful_steps_in_window,
            "window_optimizer_attempts": attempted_steps_in_window,
            "window_skipped_optimizer_steps": (
                attempted_steps_in_window - successful_steps_in_window
            ),
            "window_elapsed_seconds": window_elapsed,
            "cells_per_second": window["cell_count"] / window_elapsed,
            "window_cells_per_second": window["cell_count"] / window_elapsed,
            # The pre-clipping norm from the last successful optimizer step.
            # It is null if a partial window contains only skipped FP16 steps.
            "grad_norm": safe_gradient_norm,
            "gradient_clip_rate": clip_rate,
            "learning_rate": optimizer.param_groups[0]["lr"],
            # Retain the concise historical alias for existing log readers.
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(metrics_path, progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        window_metrics = MetricAccumulator()
        successful_steps_in_window = 0
        attempted_steps_in_window = 0
        clipped_steps_in_window = 0
        last_gradient_norm = None
        window_start = time.monotonic()

    for batch_index, clean_expression in enumerate(loader):
        microbatch_cells = int(clean_expression.shape[0])
        clean_expression = clean_expression.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_stop = min(group_start + accumulation_steps, num_batches)
        nominal_batch_size = loader.batch_size
        if not isinstance(nominal_batch_size, int):
            raise RuntimeError("Training DataLoader must have a fixed batch size.")
        group_cells = nominal_batch_size * (group_stop - group_start)
        if group_stop == num_batches:
            final_batch_cells = len(loader.dataset) - nominal_batch_size * (
                num_batches - 1
            )
            group_cells += final_batch_cells - nominal_batch_size
        if group_cells <= 0:
            raise RuntimeError("Gradient-accumulation group contains no cells.")

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            output = runner(clean_expression, generator=diffusion_generator)
            # Every model loss is normalized by its local B*G. Weighting by the
            # number of cells keeps the accumulated gradient normalized by the
            # full group's cell-gene count, including a smaller final batch.
            loss = output.loss * (microbatch_cells / group_cells)
        # ``loss`` is ``weighted_nll_sum / normalizer`` with a positive integer
        # normalizer, so the host-side sum decides finiteness.  Reusing the
        # single transfer above avoids a second synchronizing copy per batch.
        microbatch_metrics = MetricAccumulator.from_output(output)
        if not math.isfinite(microbatch_metrics.weighted_nll_sum):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}."
            )
        metrics.merge(microbatch_metrics)
        accumulation_group_metrics.merge(microbatch_metrics)
        scaler.scale(loss).backward()

        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == num_batches
        )
        if not should_step:
            continue

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )
        gradient_norm_value: Optional[float] = None
        if not scaler.is_enabled():
            gradient_norm_value = float(gradient_norm.detach().item())
            if not math.isfinite(gradient_norm_value):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_was_skipped = scaler.get_scale() < previous_scale
        # The loss was evaluated even when FP16 overflow causes the optimizer
        # update to be skipped, so its sufficient statistics belong to the
        # current monitoring window. Only successful updates close an interval.
        window_metrics.merge(accumulation_group_metrics)
        attempted_steps_in_window += 1
        if not optimizer_was_skipped:
            if gradient_norm_value is None:
                gradient_norm_value = float(gradient_norm.detach().item())
            if not math.isfinite(gradient_norm_value):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )
            scheduler.step()
            global_step += 1
            successful_steps_in_window += 1
            clipped_steps_in_window += int(gradient_norm_value > max_grad_norm)
            last_gradient_norm = gradient_norm_value

        accumulation_group_metrics = MetricAccumulator()

        if successful_steps_in_window == log_every:
            flush_progress_window(
                batch_number=batch_index + 1,
                reason="interval",
            )

        if STOP_REQUEST.requested:
            flush_progress_window(
                batch_number=batch_index + 1,
                reason="interrupted",
            )
            result = metrics.result()
            result["elapsed_seconds"] = time.monotonic() - epoch_start
            return result, global_step, True

    flush_progress_window(batch_number=num_batches, reason="epoch_end")
    result = metrics.result()
    result["elapsed_seconds"] = time.monotonic() - epoch_start
    return result, global_step, False


@torch.no_grad()
def validate(
    *,
    runner: Any,
    model: MaskedDiffusionTrainingModule,
    loader: DataLoader[Tensor],
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    seed: int,
) -> tuple[dict[str, Any], bool]:
    model.eval()
    metrics = MetricAccumulator()
    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(seed)
    start = time.monotonic()
    for clean_expression in loader:
        clean_expression = clean_expression.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            output = runner(clean_expression, generator=validation_generator)
        microbatch_metrics = MetricAccumulator.from_output(output)
        if not math.isfinite(microbatch_metrics.weighted_nll_sum):
            raise FloatingPointError("Validation produced a non-finite loss.")
        metrics.merge(microbatch_metrics)
        if STOP_REQUEST.requested:
            result = metrics.result()
            result[PRIMARY_VALIDATION_METRIC] = result["loss"]
            result["elapsed_seconds"] = time.monotonic() - start
            return result, True
    result = metrics.result()
    result[PRIMARY_VALIDATION_METRIC] = result["loss"]
    result["elapsed_seconds"] = time.monotonic() - start
    return result, False


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    args.data_path = resolve_input(args.data_path, "Data h5ad")
    args.gene_mapping_path = resolve_input(args.gene_mapping_path, "Gene mapping")
    args.gene_weights_path = resolve_input(args.gene_weights_path, "Gene weights")
    args.gene_manifest_path = resolve_input(
        args.gene_manifest_path,
        "Gene embedding manifest",
    )
    if args.resume is not None:
        args.resume = resolve_input(args.resume, "Resume checkpoint")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    run_config_path = args.output_dir / "run_config.json"
    if args.resume is None and (
        run_config_path.exists()
        or metrics_path.exists()
        or any(checkpoint_dir.glob("*.pt"))
    ):
        raise FileExistsError(
            "Output directory already contains a training run; choose a new "
            "--output-dir or provide --resume."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires one CUDA GPU; none is available.")
    if torch.cuda.device_count() != 1:
        print(
            f"WARNING: {torch.cuda.device_count()} GPUs are visible; only "
            "cuda:0 is used.",
            flush=True,
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_global_seed(args.seed)
    autocast_enabled, autocast_dtype = amp_settings(args.precision)

    expected_genes = read_expected_genes(
        args.gene_mapping_path,
        index_column=args.gene_index_column,
        name_column=args.gene_name_column,
    )
    metadata = ad.read_h5ad(args.data_path, backed="r")
    try:
        n_obs = int(metadata.n_obs)
    finally:
        close_backed(metadata)
    train_indices, val_indices = deterministic_split(
        n_obs,
        args.val_fraction,
        args.seed,
        args.max_train_cells,
        args.max_val_cells,
    )
    train_dataset = BackedH5adRowDataset(
        args.data_path,
        args.matrix_key,
        train_indices,
        expected_genes,
    )
    val_dataset = BackedH5adRowDataset(
        args.data_path,
        args.matrix_key,
        val_indices,
        expected_genes,
    )
    sampler_generator = torch.Generator(device="cpu")
    train_worker_generator = torch.Generator(device="cpu")
    train_worker_generator.manual_seed(args.seed + 200_000)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        sampler_generator=sampler_generator,
        worker_generator=train_worker_generator,
    )
    validation_workers = min(args.num_workers, 2)
    validation_worker_generator = torch.Generator(device="cpu")
    validation_worker_generator.manual_seed(args.seed + 300_000)
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=validation_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        sampler_generator=None,
        worker_generator=validation_worker_generator,
    )

    data_stat = args.data_path.stat()
    data_contract = {
        "path": str(args.data_path),
        "file_size": data_stat.st_size,
        "file_mtime_ns": data_stat.st_mtime_ns,
        "matrix_key": args.matrix_key,
        "matrix_dtype": train_dataset.matrix_dtype,
        "n_obs": n_obs,
        "n_vars": NUM_GENES,
        "train_cells": len(train_dataset),
        "validation_cells": len(val_dataset),
        "gene_order_sha256": sha256_strings(expected_genes),
        "split_sha256": sha256_indices(train_indices, val_indices),
    }

    model_config = build_model_config(args)
    model = MaskedDiffusionTrainingModule.from_config(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    try:
        optimizer = AdamW(model.parameters(), fused=True, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    updates_per_epoch = math.ceil(
        len(train_loader) / args.grad_accumulation_steps
    )
    total_steps = args.epochs * updates_per_epoch
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    diffusion_generator = torch.Generator(device=device)
    diffusion_generator.manual_seed(args.seed + 1)
    validation_seed = args.seed + VALIDATION_SEED_OFFSET

    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    early_stopping_bad_validations = 0
    last_checkpoint_current_epoch = -1
    last_checkpoint_epoch_completed = True
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        validate_resume_checkpoint(
            checkpoint,
            model_config=model_config,
            args=args,
            data_contract=data_contract,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"], diffusion_generator)
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_primary_validation_metric"])
        early_stopping_bad_validations = int(
            checkpoint["early_stopping_bad_validations"]
        )
        last_checkpoint_current_epoch = int(checkpoint["current_epoch"])
        last_checkpoint_epoch_completed = bool(checkpoint["epoch_completed"])
        print(
            f"Resuming from {args.resume}; next epoch index={start_epoch}, "
            f"global step={global_step}.",
            flush=True,
        )

    run_config = {
        "architecture_version": ARCHITECTURE_VERSION,
        "arguments": vars(args),
        "model_config": asdict(model_config),
        "data_contract": data_contract,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "updates_per_epoch": updates_per_epoch,
        "total_optimizer_steps": total_steps,
        "primary_validation_metric": {
            "name": PRIMARY_VALIDATION_METRIC,
            "mode": "min",
            "definition": (
                "sum(M * t^-1 * hurdle_truncated_normal_nll) / "
                f"(validation_cells * {NUM_GENES}) over the complete fixed "
                "validation split; the validation corruption generator is reset "
                "to validation_seed before every validation."
            ),
            "validation_seed": validation_seed,
        },
        "resume_semantics": (
            "completed checkpoints resume at the next epoch; interrupted "
            "checkpoints retain updates and restart the interrupted epoch"
        ),
    }
    atomic_json_dump(run_config, run_config_path)
    print(json.dumps(jsonable(run_config), sort_keys=True), flush=True)

    runner: Any = model
    if args.torch_compile:
        print("WARNING: enabling experimental torch.compile.", flush=True)
        runner = torch.compile(model)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    for epoch in range(start_epoch, args.epochs):
        if STOP_REQUEST.requested:
            metrics = {
                "event": "interrupted_before_epoch",
                "epoch": epoch + 1,
                "global_step": global_step,
                "signal": STOP_REQUEST.signal_number,
            }
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                diffusion_generator=diffusion_generator,
                model_config=model_config,
                args=args,
                data_contract=data_contract,
                current_epoch=last_checkpoint_current_epoch,
                epoch_completed=last_checkpoint_epoch_completed,
                next_epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                early_stopping_bad_validations=early_stopping_bad_validations,
                reason="interrupted_before_epoch",
                metrics=metrics,
            )
            interrupted_path = checkpoint_dir / "interrupted.pt"
            atomic_torch_save(payload, interrupted_path)
            append_jsonl(metrics_path, metrics)
            print(f"Saved interrupted checkpoint: {interrupted_path}", flush=True)
            return 130

        sampler_generator.manual_seed(args.seed + epoch)
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics, global_step, interrupted = train_one_epoch(
            runner=runner,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            diffusion_generator=diffusion_generator,
            device=device,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            accumulation_steps=args.grad_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            metrics_path=metrics_path,
            epoch=epoch,
            global_step=global_step,
        )

        if interrupted:
            metrics = {
                "event": "interrupted",
                "epoch": epoch + 1,
                "global_step": global_step,
                "signal": STOP_REQUEST.signal_number,
                "train": train_metrics,
            }
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                diffusion_generator=diffusion_generator,
                model_config=model_config,
                args=args,
                data_contract=data_contract,
                current_epoch=epoch,
                epoch_completed=False,
                next_epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                early_stopping_bad_validations=early_stopping_bad_validations,
                reason="interrupted",
                metrics=metrics,
            )
            interrupted_path = checkpoint_dir / "interrupted.pt"
            atomic_torch_save(payload, interrupted_path)
            append_jsonl(metrics_path, metrics)
            print(f"Saved interrupted checkpoint: {interrupted_path}", flush=True)
            return 130

        validation_metrics: Optional[dict[str, Any]] = None
        # A signal may arrive after train_one_epoch's final optimizer-boundary
        # check. The epoch is complete in that case, so skip validation and
        # checkpoint it with next_epoch=epoch+1.
        validation_interrupted = STOP_REQUEST.requested
        if (
            not validation_interrupted
            and args.validate_every
            and (epoch + 1) % args.validate_every == 0
        ):
            validation_metrics, validation_interrupted = validate(
                runner=runner,
                model=model,
                loader=val_loader,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                seed=validation_seed,
            )

        next_epoch = epoch + 1
        stop_requested_at_boundary = (
            validation_interrupted or STOP_REQUEST.requested
        )
        completed_validation_metric = (
            float(validation_metrics[PRIMARY_VALIDATION_METRIC])
            if validation_metrics is not None and not validation_interrupted
            else None
        )
        is_new_best = (
            completed_validation_metric is not None
            and completed_validation_metric < best_validation_loss
        )
        if completed_validation_metric is not None:
            if is_new_best:
                best_validation_loss = completed_validation_metric
                early_stopping_bad_validations = 0
            else:
                early_stopping_bad_validations += 1

        early_stopping_triggered = (
            completed_validation_metric is not None
            and args.early_stopping_patience > 0
            and next_epoch >= args.early_stopping_min_epochs
            and early_stopping_bad_validations
            >= args.early_stopping_patience
        )
        reason = (
            "interrupted_after_epoch"
            if stop_requested_at_boundary
            else "early_stopping"
            if early_stopping_triggered
            else "epoch_end"
        )
        epoch_metrics = {
            "event": "epoch_end",
            "epoch": epoch + 1,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_interrupted": validation_interrupted,
            "primary_validation_metric": PRIMARY_VALIDATION_METRIC,
            "is_new_best": is_new_best,
            "early_stopping_bad_validations": early_stopping_bad_validations,
            "early_stopping_triggered": early_stopping_triggered,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        append_jsonl(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            diffusion_generator=diffusion_generator,
            model_config=model_config,
            args=args,
            data_contract=data_contract,
            current_epoch=epoch,
            epoch_completed=True,
            next_epoch=next_epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            early_stopping_bad_validations=early_stopping_bad_validations,
            reason=reason,
            metrics=epoch_metrics,
        )
        atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            atomic_torch_save(
                payload,
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
            )

        if is_new_best:
            atomic_torch_save(payload, checkpoint_dir / "best.pt")

        # Re-read the global flag after checkpoint writes. This also catches a
        # termination request that arrived while latest/best/archive was being
        # serialized. Re-save latest with explicit interruption metadata in
        # that narrow case, then create the stable resume target.
        if stop_requested_at_boundary or STOP_REQUEST.requested:
            if not stop_requested_at_boundary:
                epoch_metrics["stop_requested_after_checkpoint"] = True
                payload["reason"] = "interrupted_after_epoch"
                payload["metrics"] = epoch_metrics
                atomic_torch_save(payload, checkpoint_dir / "latest.pt")
            interrupted_path = checkpoint_dir / "interrupted.pt"
            atomic_torch_save(payload, interrupted_path)
            print(
                "Stop requested at an epoch boundary; saved "
                f"{interrupted_path} and latest.pt.",
                flush=True,
            )
            return 130

        last_checkpoint_current_epoch = epoch
        last_checkpoint_epoch_completed = True

        if early_stopping_triggered:
            print(
                "Early stopping: "
                f"{PRIMARY_VALIDATION_METRIC} did not improve for "
                f"{early_stopping_bad_validations} consecutive completed "
                f"validations after {next_epoch} epochs.",
                flush=True,
            )
            return 0

    if STOP_REQUEST.requested:
        print(
            "Stop requested after the final epoch checkpoint; latest.pt is "
            "complete and recoverable.",
            flush=True,
        )
        return 130

    print("Training completed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
