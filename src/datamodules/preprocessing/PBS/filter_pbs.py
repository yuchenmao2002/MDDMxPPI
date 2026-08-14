#!/usr/bin/env python3
"""Filter PBS-treated cells from a large h5ad file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
RAW_PBS_DIR = PROJECT_ROOT / "data" / "raw" / "PBS"
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"

DEFAULT_INPUT = RAW_PBS_DIR / "Parse_10M_PBMC_cytokines.h5ad"
DEFAULT_OUTPUT = INTERIM_PBS_DIR / "Parse_10M_PBMC_PBS.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save cells with obs['treatment'] == PBS as a new h5ad file."
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
        help=f"Output h5ad file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--treatment",
        default="PBS",
        help="Treatment value to keep from obs['treatment']. Default: PBS",
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
        print(f"Reading input in backed mode: {input_path}")
        adata_backed = ad.read_h5ad(input_path, backed="r")

        if "treatment" not in adata_backed.obs.columns:
            print("Column obs['treatment'] was not found in the input h5ad.", file=sys.stderr)
            return 1

        mask = adata_backed.obs["treatment"].astype(str) == args.treatment
        n_total = adata_backed.n_obs
        n_selected = int(mask.sum())

        print(f"Total cells: {n_total}")
        print(f"Selected {args.treatment} cells: {n_selected}")

        if n_selected == 0:
            print(f"No cells matched treatment == {args.treatment!r}.", file=sys.stderr)
            return 1

        subset = adata_backed[mask.to_numpy(), :]

        print("Loading selected cells into memory...")
        subset_memory = subset.to_memory()

        print(f"Writing output: {output_path}")
        subset_memory.write_h5ad(output_path)
    finally:
        if adata_backed is not None and adata_backed.isbacked:
            adata_backed.file.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
