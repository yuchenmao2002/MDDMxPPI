#!/usr/bin/env python3
"""Classify PBS matches for HGNC missing symbols using queried HGNC info."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"

DEFAULT_HGNC_INPUT = INTERIM_PBS_DIR / "hgnc_missing_with_info.csv"
DEFAULT_PBS_INPUT = INTERIM_PBS_DIR / "pbs_missing.csv"
DEFAULT_MAPPING_OUTPUT = INTERIM_PBS_DIR / "missing_match.csv"
DEFAULT_CONFLICTS_OUTPUT = INTERIM_PBS_DIR / "missing_match_conflicts.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use HGNC Previous, Ensembl, and Alias values to match hgnc_missing "
            "symbols against pbs_missing indices. Unique Previous/Ensembl matches "
            "are written to a mapping file; Alias matches and multi-hit rows are "
            "written to a conflicts file for manual review."
        )
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=(
            "HGNC CSV with Previous, Alias, and Ensembl columns. "
            f"Default: {DEFAULT_HGNC_INPUT}"
        ),
    )
    parser.add_argument(
        "--pbs-input",
        type=Path,
        default=DEFAULT_PBS_INPUT,
        help=f"PBS missing-index CSV. Default: {DEFAULT_PBS_INPUT}",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT,
        help=(
            "Output CSV for unique Previous/Ensembl matches. "
            f"Default: {DEFAULT_MAPPING_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default=DEFAULT_CONFLICTS_OUTPUT,
        help=(
            "Output CSV for Alias matches and rows with multiple possible matches. "
            f"Default: {DEFAULT_CONFLICTS_OUTPUT}"
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        return reader.fieldnames, list(reader)


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


def split_values(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def match_values(
    row: dict[str, str],
    column: str,
    pbs_indices: set[str],
) -> list[str]:
    return [value for value in split_values(row.get(column, "")) if value in pbs_indices]


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
        mapping_output = args.mapping_output.expanduser().absolute()
        conflicts_output = args.conflicts_output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        pbs_fieldnames, pbs_rows = read_csv_rows(pbs_input)

        previous_col = find_column(hgnc_fieldnames, ("Previous", "previous"), hgnc_input.name)
        alias_col = find_column(hgnc_fieldnames, ("Alias", "alias"), hgnc_input.name)
        ensembl_col = find_column(hgnc_fieldnames, ("Ensembl", "ensembl"), hgnc_input.name)
        symbol_col = find_column(hgnc_fieldnames, ("Symbol", "symbol"), hgnc_input.name)
        pbs_index_col = find_column(pbs_fieldnames, ("Index", "index"), pbs_input.name)

        pbs_indices = {row[pbs_index_col] for row in pbs_rows}
        mapping_rows: list[dict[str, str]] = []
        conflict_rows: list[dict[str, str]] = []
        unmatched_rows = 0

        for row in hgnc_rows:
            symbol = row[symbol_col]
            previous_matches = match_values(row, previous_col, pbs_indices)
            ensembl_matches = match_values(row, ensembl_col, pbs_indices)
            alias_matches = match_values(row, alias_col, pbs_indices)
            non_alias_matches = unique_in_order(previous_matches + ensembl_matches)
            all_matches = unique_in_order(non_alias_matches + alias_matches)

            if not all_matches:
                unmatched_rows += 1
                continue

            if alias_matches or len(all_matches) > 1:
                conflict_rows.append(
                    {
                        "Symbol": symbol,
                        "potential_PBS_Index": "; ".join(all_matches),
                        "PBS_Index": "",
                    }
                )
                continue

            mapping_rows.append(
                {
                    "Symbol": symbol,
                    "PBS_Index": non_alias_matches[0],
                }
            )

        write_csv_rows(mapping_output, ["Symbol", "PBS_Index"], mapping_rows)
        write_csv_rows(
            conflicts_output,
            ["Symbol", "potential_PBS_Index", "PBS_Index"],
            conflict_rows,
        )

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"PBS missing indices: {len(pbs_indices):,}")
        print(f"Unique Previous/Ensembl mappings: {len(mapping_rows):,}")
        print(f"Conflicts for manual review: {len(conflict_rows):,}")
        print(f"Unmatched rows: {unmatched_rows:,}")
        print(f"Mapping output: {mapping_output}")
        print(f"Conflicts output: {conflicts_output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
