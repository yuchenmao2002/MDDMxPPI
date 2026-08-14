#!/usr/bin/env python3
"""Export gene metadata from the PBS h5ad file with Index as the first column."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"

DEFAULT_INPUT = INTERIM_PBS_DIR / "Parse_10M_PBMC_PBS.h5ad"
DEFAULT_OUTPUT = INTERIM_PBS_DIR / "pbs_index.csv"
SYMBOL_COLUMN = "Index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export adata.var index from Parse_10M_PBMC_PBS.h5ad to CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input h5ad file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def load_anndata():
    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit(
            "This script requires the 'anndata' package. "
            "Install it in your Python environment, then rerun the script."
        ) from exc
    return ad


def unique_column_name(base_name: str, existing_columns) -> str:
    name = base_name
    suffix = 1
    while name in existing_columns:
        name = f"{base_name}_{suffix}"
        suffix += 1
    return name


def format_var_metadata(var):
    gene_metadata = var.copy()
    index_values = gene_metadata.index.astype(str)

    if SYMBOL_COLUMN in gene_metadata.columns:
        symbol_values = gene_metadata.pop(SYMBOL_COLUMN).astype(str)

        if not (index_values == symbol_values.to_numpy()).all():
            index_name = gene_metadata.index.name or "var_index"
            if index_name == SYMBOL_COLUMN:
                index_name = "var_index"
            index_name = unique_column_name(index_name, gene_metadata.columns)
            gene_metadata.insert(0, index_name, index_values)

        gene_metadata.insert(0, SYMBOL_COLUMN, symbol_values.to_numpy())
    else:
        gene_metadata.insert(0, SYMBOL_COLUMN, index_values)

    return gene_metadata


def main() -> int:
    args = parse_args()

    input_path = args.input.expanduser().absolute()
    output_path = args.output.expanduser().absolute()

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if output_path.exists() and not args.overwrite:
        print(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite to replace it, or pass --output to choose another file.",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ad = load_anndata()
    adata_backed = None

    try:
        print(f"Reading gene metadata from: {input_path}")
        adata_backed = ad.read_h5ad(input_path, backed="r")

        gene_metadata = format_var_metadata(adata_backed.var)
        print(f"Genes exported: {gene_metadata.shape[0]}")
        print(f"Columns exported: {gene_metadata.shape[1]}")

        gene_metadata.to_csv(output_path, index=False)
    finally:
        if adata_backed is not None and adata_backed.isbacked:
            adata_backed.file.close()

    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
