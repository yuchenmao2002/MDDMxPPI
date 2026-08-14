#!/usr/bin/env python3
"""Match HGNC symbols with STRING preferred names and write unmatched rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).absolute().parents[4]
RAW_STRING_DIR = PROJECT_ROOT / "data" / "raw" / "STRING"
INTERIM_STRING_DIR = PROJECT_ROOT / "data" / "interim" / "STRING"

DEFAULT_HGNC_INPUT = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"
DEFAULT_PROTEIN_INPUT = RAW_STRING_DIR / "9606.protein.info.v12.0.txt"
DEFAULT_HGNC_MISSING_OUTPUT = INTERIM_STRING_DIR / "hgnc_missing.csv"
DEFAULT_STRING_MISSING_OUTPUT = INTERIM_STRING_DIR / "string_missing.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match hgnc_symbol.csv Symbol values with STRING protein.info "
            "preferred_name values, then write unmatched rows from both inputs."
        )
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=f"HGNC indexed gene-symbol CSV. Default: {DEFAULT_HGNC_INPUT}",
    )
    parser.add_argument(
        "--protein-input",
        type=Path,
        default=DEFAULT_PROTEIN_INPUT,
        help=f"STRING protein.info TSV. Default: {DEFAULT_PROTEIN_INPUT}",
    )
    parser.add_argument(
        "--hgnc-missing-output",
        type=Path,
        default=DEFAULT_HGNC_MISSING_OUTPUT,
        help=(
            "Output CSV for HGNC rows without a matching STRING preferred_name. "
            f"Default: {DEFAULT_HGNC_MISSING_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--string-missing-output",
        type=Path,
        default=DEFAULT_STRING_MISSING_OUTPUT,
        help=(
            "Output CSV for STRING protein rows without a matching HGNC Symbol. "
            f"Default: {DEFAULT_STRING_MISSING_OUTPUT}"
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


def read_csv_rows(
    path: Path,
    delimiter: str = ",",
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a header.")
        return reader.fieldnames, list(reader)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        protein_input = validate_input(args.protein_input, "Protein input")
        hgnc_missing_output = args.hgnc_missing_output.expanduser().absolute()
        string_missing_output = args.string_missing_output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        protein_fieldnames, protein_rows = read_csv_rows(
            protein_input,
            delimiter="\t",
        )

        hgnc_symbol_col = find_column(
            hgnc_fieldnames,
            ("Symbol", "symbol"),
            hgnc_input.name,
        )
        preferred_name_col = find_column(
            protein_fieldnames,
            ("preferred_name",),
            protein_input.name,
        )

        hgnc_symbols = {row[hgnc_symbol_col].strip() for row in hgnc_rows}
        preferred_names = {row[preferred_name_col].strip() for row in protein_rows}
        matched_values = hgnc_symbols & preferred_names

        missing_hgnc_rows = [
            row
            for row in hgnc_rows
            if row[hgnc_symbol_col].strip() not in matched_values
        ]
        missing_string_rows = [
            row
            for row in protein_rows
            if row[preferred_name_col].strip() not in matched_values
        ]

        write_csv_rows(hgnc_missing_output, hgnc_fieldnames, missing_hgnc_rows)
        write_csv_rows(string_missing_output, protein_fieldnames, missing_string_rows)

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"HGNC unique symbols: {len(hgnc_symbols):,}")
        print(f"STRING protein rows: {len(protein_rows):,}")
        print(f"STRING unique preferred names: {len(preferred_names):,}")
        print(f"Matched values: {len(matched_values):,}")
        print(f"Missing HGNC rows: {len(missing_hgnc_rows):,}")
        print(f"Missing STRING rows: {len(missing_string_rows):,}")
        print(f"HGNC missing output: {hgnc_missing_output}")
        print(f"STRING missing output: {string_missing_output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
