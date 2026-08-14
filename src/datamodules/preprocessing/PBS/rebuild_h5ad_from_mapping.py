#!/usr/bin/env python3
"""Rebuild the PBS h5ad gene axis from hgnc_pbs_mapping.csv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


# HDF5 file locking can hang on the cluster network filesystem when opening h5ad
# files in backed mode. Set this before importing anndata/h5py.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"
PROCESSED_PBS_DIR = PROJECT_ROOT / "data" / "processed" / "PBS"

DEFAULT_INPUT = INTERIM_PBS_DIR / "Parse_10M_PBMC_PBS.h5ad"
DEFAULT_MAPPING = PROCESSED_PBS_DIR / "hgnc_pbs_mapping.csv"
DEFAULT_OUTPUT = INTERIM_PBS_DIR / "Parse_10M_PBMC_PBS_ordered.h5ad"
DEFAULT_PLACEHOLDER = "__NO_PBS_INDEX__"
REQUIRED_MAPPING_COLUMNS = ("Symbol", "Index", "PBS_Index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new h5ad whose gene axis follows hgnc_pbs_mapping.csv. "
            "Mapped genes are copied from the PBS h5ad, missing genes are filled with 0, "
            "and unmapped source genes are discarded."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input PBS h5ad file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help=f"Symbol/PBS index mapping CSV. Default: {DEFAULT_MAPPING}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output rebuilt h5ad file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--placeholder",
        default=DEFAULT_PLACEHOLDER,
        help=f"PBS_Index placeholder for missing genes. Default: {DEFAULT_PLACEHOLDER}",
    )
    parser.add_argument(
        "--multi-pbs-index",
        choices=("sum", "first", "error"),
        default="sum",
        help=(
            "How to handle rows whose PBS_Index contains multiple ';'-separated genes. "
            "'sum' adds their counts, 'first' uses the first one, and 'error' stops. "
            "Default: sum"
        ),
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
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def load_dependencies():
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy import sparse
    except ImportError as exc:
        raise SystemExit(
            "This script requires 'anndata', 'numpy', 'pandas', and 'scipy'. "
            "Install them in your Python environment, then rerun the script."
        ) from exc

    return ad, np, pd, sparse


def validate_input(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def validate_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().absolute()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            "Use --overwrite to replace it, or pass --output to choose another file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_mapping(mapping_path: Path, pd):
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    missing_columns = [
        column for column in REQUIRED_MAPPING_COLUMNS if column not in mapping.columns
    ]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Mapping file is missing required columns: {joined}")

    for column in REQUIRED_MAPPING_COLUMNS:
        mapping[column] = mapping[column].astype(str).str.strip()

    if mapping.empty:
        raise ValueError(f"Mapping file contains no rows: {mapping_path}")

    for column in ("Symbol", "Index"):
        empty_mask = mapping[column] == ""
        if empty_mask.any():
            first_row = int(empty_mask[empty_mask].index[0]) + 2
            raise ValueError(f"Mapping column {column!r} has an empty value at CSV row {first_row}")

    duplicated_index = mapping["Index"].duplicated(keep=False)
    if duplicated_index.any():
        examples = ", ".join(mapping.loc[duplicated_index, "Index"].head(10).tolist())
        raise ValueError(f"Mapping Index values must be unique. Examples: {examples}")

    duplicated_symbol = mapping["Symbol"].duplicated(keep=False)
    if duplicated_symbol.any():
        examples = ", ".join(mapping.loc[duplicated_symbol, "Symbol"].head(10).tolist())
        raise ValueError(f"Mapping Symbol values must be unique. Examples: {examples}")

    return mapping


def split_pbs_index(value: str, placeholder: str) -> list[str]:
    if not value or value == placeholder:
        return []

    names = []
    seen = set()
    for item in value.split(";"):
        name = item.strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)

    return names


def build_source_position(var_names: list[str]) -> dict[str, int]:
    source_position: dict[str, int] = {}
    duplicate_names = set()

    for position, name in enumerate(var_names):
        if name in source_position:
            duplicate_names.add(name)
        source_position[name] = position

    if duplicate_names:
        examples = ", ".join(sorted(duplicate_names)[:10])
        raise ValueError(f"Input h5ad var_names are not unique. Examples: {examples}")

    return source_position


def build_projection_plan(
    mapping,
    source_position: dict[str, int],
    placeholder: str,
    multi_pbs_index: str,
) -> tuple[list[int], list[int], dict[str, int]]:
    source_rows: list[int] = []
    output_columns: list[int] = []
    missing_output_genes = 0
    multi_index_rows = 0
    not_found: list[str] = []

    for output_column, row in enumerate(mapping.itertuples(index=False)):
        pbs_index = getattr(row, "PBS_Index")
        pbs_names = split_pbs_index(pbs_index, placeholder)

        if not pbs_names:
            missing_output_genes += 1
            continue

        if len(pbs_names) > 1:
            multi_index_rows += 1
            if multi_pbs_index == "error":
                symbol = getattr(row, "Symbol")
                raise ValueError(
                    f"Mapping row for Symbol={symbol!r} has multiple PBS_Index values: {pbs_index}"
                )
            if multi_pbs_index == "first":
                pbs_names = pbs_names[:1]

        for pbs_name in pbs_names:
            source_row = source_position.get(pbs_name)
            if source_row is None:
                not_found.append(pbs_name)
                continue
            source_rows.append(source_row)
            output_columns.append(output_column)

    if not_found:
        unique_not_found = sorted(set(not_found))
        examples = ", ".join(unique_not_found[:20])
        raise ValueError(
            "Some PBS_Index values were not found in input h5ad var_names. "
            f"Examples: {examples}"
        )

    summary = {
        "mapped_entries": len(source_rows),
        "unique_source_genes": len(set(source_rows)),
        "missing_output_genes": missing_output_genes,
        "multi_index_rows": multi_index_rows,
    }
    return source_rows, output_columns, summary


def build_new_var(mapping):
    new_var = mapping.drop(columns=["Symbol"]).copy()
    new_var.index = mapping["Symbol"].astype(str)
    new_var.index.name = "Symbol"
    return new_var


def rebuild_sparse_x(X, source_rows, output_columns, n_source_genes, n_output_genes, np, sparse):
    projection_dtype = X.dtype if np.issubdtype(X.dtype, np.number) else np.float32
    projection_values = np.ones(len(source_rows), dtype=projection_dtype)
    projection = sparse.csc_matrix(
        (projection_values, (source_rows, output_columns)),
        shape=(n_source_genes, n_output_genes),
    )
    rebuilt = X @ projection
    return rebuilt.asformat("csr")


def rebuild_dense_x(X, source_rows, output_columns, n_output_genes, np):
    rebuilt = np.zeros((X.shape[0], n_output_genes), dtype=X.dtype)
    for source_row, output_column in zip(source_rows, output_columns):
        rebuilt[:, output_column] += X[:, source_row]
    return rebuilt


def subset_source_rows(source_rows: list[int]) -> tuple[list[int], list[int]]:
    unique_source_rows = sorted(set(source_rows))
    source_row_to_subset = {
        source_row: subset_row for subset_row, source_row in enumerate(unique_source_rows)
    }
    remapped_source_rows = [source_row_to_subset[source_row] for source_row in source_rows]
    return unique_source_rows, remapped_source_rows


def main() -> int:
    args = parse_args()

    try:
        input_path = validate_input(args.input, "Input h5ad")
        mapping_path = validate_input(args.mapping, "Mapping CSV")
        output_path = validate_output(args.output, args.overwrite)

        ad, np, pd, sparse = load_dependencies()

        print(f"Reading mapping: {mapping_path}", flush=True)
        mapping = read_mapping(mapping_path, pd)
        new_var = build_new_var(mapping)

        print(f"Reading input h5ad in backed mode: {input_path}", flush=True)
        adata_backed = ad.read_h5ad(input_path, backed="r")

        try:
            source_var_names = [str(name) for name in adata_backed.var_names]
            source_position = build_source_position(source_var_names)
            source_rows, output_columns, summary = build_projection_plan(
                mapping=mapping,
                source_position=source_position,
                placeholder=args.placeholder,
                multi_pbs_index=args.multi_pbs_index,
            )

            print(f"Input cells: {adata_backed.n_obs:,}", flush=True)
            print(f"Input genes: {adata_backed.n_vars:,}", flush=True)
            print(f"Output genes: {len(mapping):,}", flush=True)
            print(f"Mapped expression entries: {summary['mapped_entries']:,}", flush=True)
            print(f"Unique source genes loaded: {summary['unique_source_genes']:,}", flush=True)
            print(f"Zero-filled output genes: {summary['missing_output_genes']:,}", flush=True)
            print(
                f"Rows with multiple PBS_Index values: {summary['multi_index_rows']:,}",
                flush=True,
            )

            unique_source_rows, source_rows = subset_source_rows(source_rows)

            print("Loading mapped source genes into memory...", flush=True)
            adata = adata_backed[:, unique_source_rows].to_memory()
        finally:
            if adata_backed.isbacked:
                adata_backed.file.close()

        print("Rebuilding expression matrix...", flush=True)
        if sparse.issparse(adata.X):
            rebuilt_x = rebuild_sparse_x(
                X=adata.X,
                source_rows=source_rows,
                output_columns=output_columns,
                n_source_genes=adata.n_vars,
                n_output_genes=len(mapping),
                np=np,
                sparse=sparse,
            )
        else:
            rebuilt_x = rebuild_dense_x(
                X=np.asarray(adata.X),
                source_rows=source_rows,
                output_columns=output_columns,
                n_output_genes=len(mapping),
                np=np,
            )

        rebuilt_adata = ad.AnnData(
            X=rebuilt_x,
            obs=adata.obs.copy(),
            var=new_var,
        )

        write_kwargs = {}
        if args.compression != "none":
            write_kwargs["compression"] = args.compression

        print(f"Writing output h5ad: {output_path}", flush=True)
        rebuilt_adata.write_h5ad(output_path, **write_kwargs)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
