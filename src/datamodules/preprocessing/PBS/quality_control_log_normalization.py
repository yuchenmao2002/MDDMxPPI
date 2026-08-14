#!/usr/bin/env python3
"""Run cell quality control, Scrublet doublet removal, and log normalization."""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
import sys


# HDF5 file locking can hang on the cluster network filesystem when opening h5ad
# files in backed mode. Set this before importing scanpy/anndata/h5py.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"
PROCESSED_PBS_DIR = PROJECT_ROOT / "data" / "processed" / "PBS"
DEFAULT_TMPDIR = PROJECT_ROOT / ".tmp"

DEFAULT_INPUT = INTERIM_PBS_DIR / "Parse_10M_PBMC_PBS_ordered.h5ad"
DEFAULT_REPORT = INTERIM_PBS_DIR / "quality_control_log_normalization.txt"
DEFAULT_QC_OUTPUT = PROCESSED_PBS_DIR / "Parse_10M_PBMC_PBS_qc.h5ad"
DEFAULT_LN_OUTPUT = PROCESSED_PBS_DIR / "Parse_10M_PBMC_PBS_ln.h5ad"

GENE_COUNT_COLUMN = "gene_count"
TSCP_COUNT_COLUMN = "tscp_count"
MT_PERCENT_COLUMN = "pct_counts_MT"


@dataclass(frozen=True)
class QcStep:
    name: str
    before: int
    removed: int
    remaining: int


@dataclass(frozen=True)
class ScrubletBatch:
    name: str
    before: int
    removed: int
    remaining: int


def parse_scrublet_threshold(value: str) -> float | None:
    if value.lower() == "auto":
        return None

    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Scrublet threshold must be a number, or 'auto'."
        ) from exc


def format_threshold(value: float | None) -> str:
    if value is None:
        return "auto"
    return f"{value:g}"


def parse_optional_bool(value: str) -> bool | None:
    value = value.lower()
    if value == "auto":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Value must be one of: true, false, auto.")


def format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "auto"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Parse_10M_PBMC_PBS_ordered.h5ad by QC metrics, remove Scrublet "
            "doublets, write a QC h5ad, log-normalize all remaining cells, write a "
            "log-normalized h5ad, and write a QC report."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input ordered h5ad file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        default=DEFAULT_QC_OUTPUT,
        help=f"Output QC-filtered h5ad before log-normalization. Default: {DEFAULT_QC_OUTPUT}",
    )
    parser.add_argument(
        "--ln-output",
        type=Path,
        default=DEFAULT_LN_OUTPUT,
        help=f"Output QC-filtered and log-normalized h5ad. Default: {DEFAULT_LN_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"QC report TXT path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--min-gene-count",
        type=float,
        default=200.0,
        help="Keep cells with obs['gene_count'] >= this value. Default: 200",
    )
    parser.add_argument(
        "--min-tscp-count",
        type=float,
        default=1000.0,
        help="Keep cells with obs['tscp_count'] >= this value. Default: 1000",
    )
    parser.add_argument(
        "--max-tscp-count",
        type=float,
        default=100000.0,
        help="Keep cells with obs['tscp_count'] <= this value. Default: 100000",
    )
    parser.add_argument(
        "--max-pct-counts-mt",
        type=float,
        default=20.0,
        help="Keep cells with obs['pct_counts_MT'] <= this value. Default: 20",
    )
    parser.add_argument(
        "--expected-doublet-rate",
        type=float,
        default=0.06,
        help="Scrublet expected_doublet_rate. Default: 0.06",
    )
    parser.add_argument(
        "--n-prin-comps",
        type=int,
        default=30,
        help="Scrublet n_prin_comps. Default: 30",
    )
    parser.add_argument(
        "--sim-doublet-ratio",
        type=float,
        default=2.0,
        help="Scrublet sim_doublet_ratio. Default: 2.0",
    )
    parser.add_argument(
        "--scrublet-threshold",
        type=parse_scrublet_threshold,
        default=0.25,
        help=(
            "Scrublet score threshold used to call doublets. Use 'auto' for Scanpy's "
            "automatic threshold, which requires scikit-image. Default: 0.25"
        ),
    )
    parser.add_argument(
        "--scrublet-batch-key",
        default="sample",
        help=(
            "obs column used to run Scrublet separately per batch/sample. "
            "Use an empty string to disable batching. Default: sample"
        ),
    )
    parser.add_argument(
        "--scrublet-mean-center",
        action="store_true",
        help=(
            "Use Scrublet mean_center=True. By default this is disabled so Scrublet "
            "uses a sparse-friendly TruncatedSVD path."
        ),
    )
    parser.add_argument(
        "--scrublet-use-approx-neighbors",
        type=parse_optional_bool,
        default=True,
        help=(
            "Scrublet use_approx_neighbors setting: true, false, or auto. "
            "Default: true"
        ),
    )
    parser.add_argument(
        "--scrublet-max-batch-cells",
        type=int,
        default=50000,
        help=(
            "Maximum cells per Scrublet run. Batches larger than this are split "
            "into evenly sized chunks. Use 0 to disable chunking. Default: 50000"
        ),
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=10000.0,
        help="Target count sum per cell before log1p normalization. Default: 10000",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="Compression passed to AnnData.write_h5ad. Default: none",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output/report files if they already exist.",
    )
    return parser.parse_args()


def load_dependencies():
    try:
        import numpy as np
        import pandas as pd
        import scanpy as sc
    except ImportError as exc:
        raise SystemExit(
            "This script requires 'scanpy', 'numpy', and 'pandas'. "
            "Install them in your Python environment, then rerun the script."
        ) from exc

    return np, pd, sc


def validate_input(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def validate_output(path: Path, overwrite: bool, label: str) -> Path:
    path = path.expanduser().absolute()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{label} already exists: {path}\n"
            "Use --overwrite to replace it, or pass a different path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def require_obs_columns(adata, columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in columns if column not in adata.obs.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Input h5ad obs is missing required columns: {joined}")


def numeric_obs_column(obs, column: str, pd):
    values = pd.to_numeric(obs[column], errors="coerce")
    missing_count = int(values.isna().sum())
    if missing_count:
        raise ValueError(
            f"obs[{column!r}] contains {missing_count:,} missing or non-numeric values."
        )
    return values


def apply_obs_filter(obs, mask, step_name: str) -> tuple[object, QcStep]:
    before = len(obs)
    kept = int(mask.sum())
    removed = before - kept
    filtered = obs.loc[mask.to_numpy()].copy()
    step = QcStep(
        name=step_name,
        before=before,
        removed=removed,
        remaining=len(filtered),
    )
    return filtered, step


def run_scrublet_on_adata(adata, args) -> None:
    import scanpy as sc

    sc.pp.scrublet(
        adata,
        expected_doublet_rate=args.expected_doublet_rate,
        n_prin_comps=args.n_prin_comps,
        sim_doublet_ratio=args.sim_doublet_ratio,
        threshold=args.scrublet_threshold,
        mean_center=args.scrublet_mean_center,
        use_approx_neighbors=args.scrublet_use_approx_neighbors,
    )

    if "predicted_doublet" not in adata.obs.columns:
        raise ValueError("Scrublet finished but obs['predicted_doublet'] was not created.")


def run_scrublet(
    adata_backed,
    qc_obs,
    qc_positions,
    args,
    np,
) -> tuple[object, QcStep, list[ScrubletBatch]]:
    before = len(qc_positions)
    scrublet_keep_mask = np.zeros(adata_backed.n_obs, dtype=bool)
    batch_summaries: list[ScrubletBatch] = []

    if args.scrublet_batch_key:
        grouped = qc_obs.groupby(args.scrublet_batch_key, sort=True, observed=True)
        base_groups = [
            (str(name), group["_source_position"].to_numpy())
            for name, group in grouped
        ]
    else:
        base_groups = [("(all)", qc_obs["_source_position"].to_numpy())]

    groups = []
    for group_name, group_positions in base_groups:
        if args.scrublet_max_batch_cells and len(group_positions) > args.scrublet_max_batch_cells:
            n_chunks = int(np.ceil(len(group_positions) / args.scrublet_max_batch_cells))
            for chunk_number, chunk_positions in enumerate(
                np.array_split(group_positions, n_chunks),
                start=1,
            ):
                groups.append(
                    (
                        f"{group_name}__chunk_{chunk_number}_of_{n_chunks}",
                        chunk_positions,
                    )
                )
        else:
            groups.append((group_name, group_positions))

    for batch_number, (batch_name, batch_positions) in enumerate(groups, start=1):
        print(
            f"Running Scrublet batch {batch_number}/{len(groups)} "
            f"{batch_name}: {len(batch_positions):,} cells",
            flush=True,
        )

        batch_adata = adata_backed[batch_positions, :].to_memory()
        run_scrublet_on_adata(batch_adata, args)

        batch_keep = ~batch_adata.obs["predicted_doublet"].astype(bool).to_numpy()
        batch_removed = int((~batch_keep).sum())
        batch_remaining = int(batch_keep.sum())
        scrublet_keep_mask[batch_positions[batch_keep]] = True

        batch_summaries.append(
            ScrubletBatch(
                name=batch_name,
                before=len(batch_positions),
                removed=batch_removed,
                remaining=batch_remaining,
            )
        )
        print(
            f"Scrublet batch {batch_name}: removed {batch_removed:,}, "
            f"remaining {batch_remaining:,}",
            flush=True,
        )

        del batch_adata
        gc.collect()

    final_positions = qc_positions[scrublet_keep_mask[qc_positions]]
    removed = before - len(final_positions)
    step = QcStep(
        name="Scrublet predicted_doublet == False",
        before=before,
        removed=removed,
        remaining=len(final_positions),
    )
    return final_positions, step, batch_summaries


def write_report(
    path: Path,
    input_path: Path,
    qc_output_path: Path,
    ln_output_path: Path,
    args,
    steps: list[QcStep],
    scrublet_batches: list[ScrubletBatch],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        target.write("Quality control and log-normalization report\n")
        target.write(f"Input: {input_path}\n")
        target.write(f"QC output: {qc_output_path}\n")
        target.write(f"Log-normalized output: {ln_output_path}\n\n")

        target.write("Parameters\n")
        target.write(f"gene_count >= {args.min_gene_count:g}\n")
        target.write(
            f"{args.min_tscp_count:g} <= tscp_count <= {args.max_tscp_count:g}\n"
        )
        target.write(f"pct_counts_MT <= {args.max_pct_counts_mt:g}\n")
        target.write(f"scrublet expected_doublet_rate = {args.expected_doublet_rate:g}\n")
        target.write(f"scrublet n_prin_comps = {args.n_prin_comps}\n")
        target.write(f"scrublet sim_doublet_ratio = {args.sim_doublet_ratio:g}\n")
        target.write(
            f"scrublet threshold = {format_threshold(args.scrublet_threshold)}\n"
        )
        target.write(f"scrublet batch_key = {args.scrublet_batch_key or '(none)'}\n")
        target.write(f"scrublet mean_center = {args.scrublet_mean_center}\n")
        target.write(
            "scrublet use_approx_neighbors = "
            f"{format_optional_bool(args.scrublet_use_approx_neighbors)}\n"
        )
        target.write(
            f"scrublet max_batch_cells = {args.scrublet_max_batch_cells}\n"
        )
        target.write(f"normalize_total target_sum = {args.target_sum:g}\n\n")

        target.write("Filtering steps\n")
        target.write("Step\tBefore\tRemoved\tRemaining\n")
        for step in steps:
            target.write(
                f"{step.name}\t{step.before}\t{step.removed}\t{step.remaining}\n"
            )

        if scrublet_batches:
            target.write("\nScrublet batches\n")
            target.write("Batch\tBefore\tRemoved\tRemaining\n")
            for batch in scrublet_batches:
                target.write(
                    f"{batch.name}\t{batch.before}\t{batch.removed}\t{batch.remaining}\n"
                )


def main() -> int:
    args = parse_args()

    try:
        DEFAULT_TMPDIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(DEFAULT_TMPDIR))

        input_path = validate_input(args.input, "Input h5ad")
        qc_output_path = validate_output(args.qc_output, args.overwrite, "QC output h5ad")
        ln_output_path = validate_output(args.ln_output, args.overwrite, "Log-normalized output h5ad")
        report_path = validate_output(args.report, args.overwrite, "Report TXT")

        np, pd, sc = load_dependencies()

        print(f"Reading input h5ad in backed mode: {input_path}", flush=True)
        adata_backed = sc.read_h5ad(input_path, backed="r")
        required_columns = [GENE_COUNT_COLUMN, TSCP_COUNT_COLUMN, MT_PERCENT_COLUMN]
        if args.scrublet_batch_key:
            required_columns.append(args.scrublet_batch_key)
        require_obs_columns(adata_backed, tuple(required_columns))

        obs = adata_backed.obs.copy()
        obs["_source_position"] = np.arange(adata_backed.n_obs)

        steps: list[QcStep] = []
        scrublet_batches: list[ScrubletBatch] = []

        gene_count = numeric_obs_column(obs, GENE_COUNT_COLUMN, pd)
        obs, step = apply_obs_filter(
            obs,
            gene_count >= args.min_gene_count,
            f"{GENE_COUNT_COLUMN} >= {args.min_gene_count:g}",
        )
        steps.append(step)
        print(f"{step.name}: removed {step.removed:,}, remaining {step.remaining:,}", flush=True)

        tscp_count = numeric_obs_column(obs, TSCP_COUNT_COLUMN, pd)
        obs, step = apply_obs_filter(
            obs,
            (tscp_count >= args.min_tscp_count) & (tscp_count <= args.max_tscp_count),
            (
                f"{args.min_tscp_count:g} <= {TSCP_COUNT_COLUMN} "
                f"<= {args.max_tscp_count:g}"
            ),
        )
        steps.append(step)
        print(f"{step.name}: removed {step.removed:,}, remaining {step.remaining:,}", flush=True)

        pct_counts_mt = numeric_obs_column(obs, MT_PERCENT_COLUMN, pd)
        obs, step = apply_obs_filter(
            obs,
            pct_counts_mt <= args.max_pct_counts_mt,
            f"{MT_PERCENT_COLUMN} <= {args.max_pct_counts_mt:g}",
        )
        steps.append(step)
        print(f"{step.name}: removed {step.removed:,}, remaining {step.remaining:,}", flush=True)

        print("Running Scrublet doublet detection...", flush=True)
        qc_positions = obs["_source_position"].to_numpy()
        final_positions, step, scrublet_batches = run_scrublet(
            adata_backed=adata_backed,
            qc_obs=obs,
            qc_positions=qc_positions,
            args=args,
            np=np,
        )
        steps.append(step)
        print(f"{step.name}: removed {step.removed:,}, remaining {step.remaining:,}", flush=True)

        print("Loading final retained cells into memory...", flush=True)
        adata = adata_backed[final_positions, :].to_memory()
        adata_backed.file.close()
        del obs
        gc.collect()

        write_kwargs = {}
        if args.compression != "none":
            write_kwargs["compression"] = args.compression

        print(f"Writing QC-filtered h5ad: {qc_output_path}", flush=True)
        adata.write_h5ad(qc_output_path, **write_kwargs)

        print("Running total-count normalization and log1p transform...", flush=True)
        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)
        adata.uns["log_normalization"] = {
            "method": "scanpy.pp.normalize_total followed by scanpy.pp.log1p",
            "target_sum": args.target_sum,
        }

        print(f"Writing log-normalized h5ad: {ln_output_path}", flush=True)
        adata.write_h5ad(ln_output_path, **write_kwargs)

        print(f"Writing QC report: {report_path}", flush=True)
        write_report(
            report_path,
            input_path,
            qc_output_path,
            ln_output_path,
            args,
            steps,
            scrublet_batches,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
