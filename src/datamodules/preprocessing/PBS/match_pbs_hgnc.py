#!/usr/bin/env python3
"""Match HGNC symbols with PBS indices and write unmatched rows from both sides."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"

DEFAULT_HGNC_INPUT = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"
DEFAULT_PBS_INPUT = INTERIM_PBS_DIR / "pbs_index.csv"
DEFAULT_HGNC_MISSING_OUTPUT = INTERIM_PBS_DIR / "hgnc_missing.csv"
DEFAULT_PBS_MISSING_OUTPUT = INTERIM_PBS_DIR / "pbs_missing.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match hgnc_symbol.csv Symbol values with pbs_index.csv Index values, "
            "then write unmatched rows from both inputs."
        )
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=f"HGNC indexed gene-symbol CSV. Default: {DEFAULT_HGNC_INPUT}",
    )
    parser.add_argument(
        "--pbs-input",
        type=Path,
        default=DEFAULT_PBS_INPUT,
        help=f"PBS index CSV. Default: {DEFAULT_PBS_INPUT}",
    )
    parser.add_argument(
        "--hgnc-missing-output",
        type=Path,
        default=DEFAULT_HGNC_MISSING_OUTPUT,
        help=(
            "Output CSV for HGNC rows without a matching PBS index. "
            f"Default: {DEFAULT_HGNC_MISSING_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--pbs-missing-output",
        type=Path,
        default=DEFAULT_PBS_MISSING_OUTPUT,
        help=(
            "Output CSV for PBS rows without a matching HGNC symbol. "
            f"Default: {DEFAULT_PBS_MISSING_OUTPUT}"
        ),
    )
    return parser.parse_args()


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


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        return reader.fieldnames, list(reader)


def validate_input(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


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
        pbs_input = validate_input(args.pbs_input, "PBS input")
        hgnc_missing_output = args.hgnc_missing_output.expanduser().absolute()
        pbs_missing_output = args.pbs_missing_output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        pbs_fieldnames, pbs_rows = read_csv_rows(pbs_input)

        hgnc_symbol_col = find_column(hgnc_fieldnames, ("Symbol", "symbol"), hgnc_input.name)
        pbs_index_col = find_column(pbs_fieldnames, ("Index", "index"), pbs_input.name)

        hgnc_symbols = {row[hgnc_symbol_col] for row in hgnc_rows}
        pbs_indices = {row[pbs_index_col] for row in pbs_rows}
        matched_values = hgnc_symbols & pbs_indices

        missing_hgnc_rows = [
            row for row in hgnc_rows if row[hgnc_symbol_col] not in matched_values
        ]
        missing_pbs_rows = [
            row for row in pbs_rows if row[pbs_index_col] not in matched_values
        ]

        write_csv_rows(hgnc_missing_output, hgnc_fieldnames, missing_hgnc_rows)
        write_csv_rows(pbs_missing_output, pbs_fieldnames, missing_pbs_rows)

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"HGNC unique symbols: {len(hgnc_symbols):,}")
        print(f"PBS index rows: {len(pbs_rows):,}")
        print(f"PBS unique indices: {len(pbs_indices):,}")
        print(f"Matched values: {len(matched_values):,}")
        print(f"Missing HGNC rows: {len(missing_hgnc_rows):,}")
        print(f"Missing PBS rows: {len(missing_pbs_rows):,}")
        print(f"HGNC missing output: {hgnc_missing_output}")
        print(f"PBS missing output: {pbs_missing_output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
