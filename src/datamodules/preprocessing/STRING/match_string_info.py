#!/usr/bin/env python3
"""Classify STRING matches for HGNC missing symbols using queried HGNC info."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_STRING_DIR = PROJECT_ROOT / "data" / "interim" / "STRING"

DEFAULT_HGNC_INPUT = INTERIM_STRING_DIR / "hgnc_missing_with_info.csv"
DEFAULT_STRING_INPUT = INTERIM_STRING_DIR / "string_missing.csv"
DEFAULT_MATCH_OUTPUT = INTERIM_STRING_DIR / "missing_match.csv"
DEFAULT_CONFLICTS_OUTPUT = INTERIM_STRING_DIR / "missing_match_conflicts.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use HGNC Previous, Ensembl, and Alias values to match hgnc_missing "
            "symbols against string_missing preferred_name values. Unique "
            "Previous/Ensembl matches are written to a match file; Alias matches "
            "and multi-hit rows are written to a conflicts file for manual review."
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
        "--string-input",
        type=Path,
        default=DEFAULT_STRING_INPUT,
        help=f"STRING missing protein CSV. Default: {DEFAULT_STRING_INPUT}",
    )
    parser.add_argument(
        "--match-output",
        type=Path,
        default=DEFAULT_MATCH_OUTPUT,
        help=(
            "Output CSV for unique Previous/Ensembl matches. "
            f"Default: {DEFAULT_MATCH_OUTPUT}"
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


def build_preferred_name_map(
    rows: list[dict[str, str]],
    preferred_name_col: str,
    protein_id_col: str,
) -> dict[str, str]:
    preferred_name_to_protein_id: dict[str, str] = {}
    duplicate_preferred_names = []

    for row in rows:
        preferred_name = row[preferred_name_col].strip()
        string_protein_id = row[protein_id_col].strip()
        if not preferred_name or not string_protein_id:
            continue
        if preferred_name in preferred_name_to_protein_id:
            duplicate_preferred_names.append(preferred_name)
        preferred_name_to_protein_id[preferred_name] = string_protein_id

    if duplicate_preferred_names:
        examples = ", ".join(sorted(set(duplicate_preferred_names))[:10])
        raise ValueError(f"Duplicate preferred_name values in STRING input: {examples}")

    return preferred_name_to_protein_id


def match_values(
    row: dict[str, str],
    column: str,
    preferred_name_to_protein_id: dict[str, str],
) -> list[str]:
    return [
        value
        for value in split_values(row.get(column, ""))
        if value in preferred_name_to_protein_id
    ]


def match_row(
    row: dict[str, str],
    symbol_col: str,
    previous_col: str,
    alias_col: str,
    ensembl_col: str,
    preferred_name_to_protein_id: dict[str, str],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    symbol = row[symbol_col].strip()
    previous_matches = match_values(row, previous_col, preferred_name_to_protein_id)
    ensembl_matches = match_values(row, ensembl_col, preferred_name_to_protein_id)
    alias_matches = match_values(row, alias_col, preferred_name_to_protein_id)
    non_alias_matches = unique_in_order(previous_matches + ensembl_matches)
    all_matches = unique_in_order(non_alias_matches + alias_matches)

    if not all_matches:
        return None, None

    if alias_matches or len(all_matches) > 1:
        return None, {
            "Symbol": symbol,
            "potential_preferred_name": "; ".join(all_matches),
            "potential_STRING_Protein_ID": "; ".join(
                preferred_name_to_protein_id[preferred_name]
                for preferred_name in all_matches
            ),
            "STRING_Protein_ID": "",
        }

    preferred_name = non_alias_matches[0]
    return {
        "Symbol": symbol,
        "matched_preferred_name": preferred_name,
        "STRING_Protein_ID": preferred_name_to_protein_id[preferred_name],
    }, None


def main() -> int:
    args = parse_args()

    try:
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        string_input = validate_input(args.string_input, "STRING input")
        match_output = args.match_output.expanduser().absolute()
        conflicts_output = args.conflicts_output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        string_fieldnames, string_rows = read_csv_rows(string_input)

        symbol_col = find_column(hgnc_fieldnames, ("Symbol", "symbol"), hgnc_input.name)
        previous_col = find_column(hgnc_fieldnames, ("Previous", "previous"), hgnc_input.name)
        alias_col = find_column(hgnc_fieldnames, ("Alias", "alias"), hgnc_input.name)
        ensembl_col = find_column(hgnc_fieldnames, ("Ensembl", "ensembl"), hgnc_input.name)
        preferred_name_col = find_column(
            string_fieldnames,
            ("preferred_name",),
            string_input.name,
        )
        protein_id_col = find_column(
            string_fieldnames,
            ("#string_protein_id", "string_protein_id"),
            string_input.name,
        )

        preferred_name_to_protein_id = build_preferred_name_map(
            string_rows,
            preferred_name_col=preferred_name_col,
            protein_id_col=protein_id_col,
        )
        match_rows: list[dict[str, str]] = []
        conflict_rows: list[dict[str, str]] = []
        unmatched_rows = 0

        for row in hgnc_rows:
            match_row_output, conflict_row = match_row(
                row,
                symbol_col=symbol_col,
                previous_col=previous_col,
                alias_col=alias_col,
                ensembl_col=ensembl_col,
                preferred_name_to_protein_id=preferred_name_to_protein_id,
            )
            if match_row_output is not None:
                match_rows.append(match_row_output)
            elif conflict_row is not None:
                conflict_rows.append(conflict_row)
            else:
                unmatched_rows += 1

        write_csv_rows(
            match_output,
            ["Symbol", "matched_preferred_name", "STRING_Protein_ID"],
            match_rows,
        )
        write_csv_rows(
            conflicts_output,
            [
                "Symbol",
                "potential_preferred_name",
                "potential_STRING_Protein_ID",
                "STRING_Protein_ID",
            ],
            conflict_rows,
        )

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"STRING missing preferred names: {len(preferred_name_to_protein_id):,}")
        print(f"Unique Previous/Ensembl matches: {len(match_rows):,}")
        print(f"Conflicts for manual review: {len(conflict_rows):,}")
        print(f"Unmatched rows: {unmatched_rows:,}")
        print(f"Match output: {match_output}")
        print(f"Conflicts output: {conflicts_output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
