#!/usr/bin/env python3
"""Match Geneformer V2 gene names/IDs with HGNC symbols and export missing rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_GENEFORMER_DIR = PROJECT_ROOT / "data" / "interim" / "Geneformer"

DEFAULT_V2_INPUT = INTERIM_GENEFORMER_DIR / "V2_gene_name_id.csv"
DEFAULT_HGNC_INPUT = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"
DEFAULT_HGNC_MISSING_OUTPUT = INTERIM_GENEFORMER_DIR / "hgnc_missing.csv"
DEFAULT_V2_MISSING_OUTPUT = INTERIM_GENEFORMER_DIR / "V2_missing.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match V2_gene_name_id.csv gene_name_or_id values with "
            "hgnc_symbol.csv Symbol values, then write unmatched rows from "
            "both inputs."
        )
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=DEFAULT_V2_INPUT,
        help=f"Geneformer V2 gene-name/ID CSV. Default: {DEFAULT_V2_INPUT}",
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=f"HGNC indexed gene-symbol CSV. Default: {DEFAULT_HGNC_INPUT}",
    )
    parser.add_argument(
        "--hgnc-missing-output",
        type=Path,
        default=DEFAULT_HGNC_MISSING_OUTPUT,
        help=(
            "Output CSV for HGNC rows without a matching V2 gene name/ID. "
            f"Default: {DEFAULT_HGNC_MISSING_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--v2-missing-output",
        type=Path,
        default=DEFAULT_V2_MISSING_OUTPUT,
        help=(
            "Output CSV for V2 rows without a matching HGNC Symbol. "
            f"Default: {DEFAULT_V2_MISSING_OUTPUT}"
        ),
    )
    return parser.parse_args()


def validate_input(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        return reader.fieldnames, list(reader)


def find_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
    table_name: str,
) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    lowered_candidates = {candidate.lower() for candidate in candidates}
    for fieldname in fieldnames:
        if fieldname.lower() in lowered_candidates:
            return fieldname

    joined = ", ".join(candidates)
    raise ValueError(f"{table_name} does not contain any of these columns: {joined}")


def write_csv_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        v2_input = validate_input(args.v2_input, "Geneformer V2 input")
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        hgnc_missing_output = args.hgnc_missing_output.expanduser().absolute()
        v2_missing_output = args.v2_missing_output.expanduser().absolute()

        v2_fieldnames, v2_rows = read_csv_rows(v2_input)
        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)

        v2_gene_name_col = find_column(
            v2_fieldnames,
            ("gene_name_or_id",),
            v2_input.name,
        )
        hgnc_symbol_col = find_column(
            hgnc_fieldnames,
            ("Symbol", "symbol"),
            hgnc_input.name,
        )

        v2_gene_names = {row[v2_gene_name_col].strip() for row in v2_rows}
        hgnc_symbols = {row[hgnc_symbol_col].strip() for row in hgnc_rows}
        matched_values = v2_gene_names & hgnc_symbols

        missing_hgnc_rows = [
            row
            for row in hgnc_rows
            if row[hgnc_symbol_col].strip() not in matched_values
        ]
        missing_v2_rows = [
            row
            for row in v2_rows
            if row[v2_gene_name_col].strip() not in matched_values
        ]

        write_csv_rows(
            hgnc_missing_output,
            hgnc_fieldnames,
            missing_hgnc_rows,
        )
        write_csv_rows(
            v2_missing_output,
            v2_fieldnames,
            missing_v2_rows,
        )

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"HGNC unique symbols: {len(hgnc_symbols):,}")
        print(f"V2 rows: {len(v2_rows):,}")
        print(f"V2 unique gene names/IDs: {len(v2_gene_names):,}")
        print(f"Matched values: {len(matched_values):,}")
        print(f"Missing HGNC rows: {len(missing_hgnc_rows):,}")
        print(f"Missing V2 rows: {len(missing_v2_rows):,}")
        print(f"HGNC missing output: {hgnc_missing_output}")
        print(f"V2 missing output: {v2_missing_output}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
