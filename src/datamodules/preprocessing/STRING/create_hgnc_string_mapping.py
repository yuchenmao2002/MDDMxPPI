#!/usr/bin/env python3
"""Create HGNC Symbol/Index to STRING protein ID mapping."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).absolute().parents[4]
RAW_STRING_DIR = PROJECT_ROOT / "data" / "raw" / "STRING"
INTERIM_STRING_DIR = PROJECT_ROOT / "data" / "interim" / "STRING"
PROCESSED_STRING_DIR = PROJECT_ROOT / "data" / "processed" / "STRING"

DEFAULT_HGNC_INPUT = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"
DEFAULT_PROTEIN_INPUT = RAW_STRING_DIR / "9606.protein.info.v12.0.txt"
DEFAULT_MISSING_MATCH_INPUT = INTERIM_STRING_DIR / "missing_match.csv"
DEFAULT_CONFLICTS_INPUT = INTERIM_STRING_DIR / "missing_mapping_conflicts.csv"
DEFAULT_OUTPUT = PROCESSED_STRING_DIR / "hgnc_string_mapping.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use hgnc_symbol.csv as the base vocabulary and add STRING_Protein_ID "
            "from direct preferred_name matches, automatic missing matches, and "
            "manually reviewed conflict matches."
        )
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=f"HGNC Symbol/Index CSV. Default: {DEFAULT_HGNC_INPUT}",
    )
    parser.add_argument(
        "--protein-input",
        type=Path,
        default=DEFAULT_PROTEIN_INPUT,
        help=f"STRING protein.info TSV. Default: {DEFAULT_PROTEIN_INPUT}",
    )
    parser.add_argument(
        "--missing-match-input",
        type=Path,
        default=DEFAULT_MISSING_MATCH_INPUT,
        help=f"Automatic STRING missing match CSV. Default: {DEFAULT_MISSING_MATCH_INPUT}",
    )
    parser.add_argument(
        "--conflicts-input",
        type=Path,
        default=DEFAULT_CONFLICTS_INPUT,
        help=(
            "Manually reviewed STRING missing conflict CSV. "
            f"Default: {DEFAULT_CONFLICTS_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Symbol/Index/STRING_Protein_ID CSV. Default: {DEFAULT_OUTPUT}",
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


def split_values(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def add_mapping(
    target: dict[str, str],
    symbol: str,
    string_protein_id: str,
    source_name: str,
) -> None:
    if not symbol or not string_protein_id:
        return
    existing = target.get(symbol)
    if existing is not None and existing != string_protein_id:
        raise ValueError(
            f"Conflicting STRING_Protein_ID values for {symbol} in {source_name}: "
            f"{existing} vs {string_protein_id}"
        )
    target[symbol] = string_protein_id


def build_direct_map(
    protein_rows: list[dict[str, str]],
    protein_id_col: str,
    preferred_name_col: str,
) -> dict[str, str]:
    direct_map: dict[str, str] = {}

    for row in protein_rows:
        preferred_name = row[preferred_name_col].strip()
        string_protein_id = row[protein_id_col].strip()
        if not preferred_name or not string_protein_id:
            continue
        add_mapping(
            direct_map,
            symbol=preferred_name,
            string_protein_id=string_protein_id,
            source_name="protein input",
        )

    return direct_map


def build_missing_match_map(
    rows: list[dict[str, str]],
    symbol_col: str,
    protein_id_col: str,
) -> dict[str, str]:
    missing_match_map: dict[str, str] = {}

    for row in rows:
        add_mapping(
            missing_match_map,
            symbol=row[symbol_col].strip(),
            string_protein_id=row[protein_id_col].strip(),
            source_name="missing match input",
        )

    return missing_match_map


def build_conflict_map(
    rows: list[dict[str, str]],
    symbol_col: str,
    protein_id_col: str,
) -> dict[str, str]:
    conflict_map: dict[str, str] = {}

    for row in rows:
        string_protein_ids = split_values(row.get(protein_id_col, ""))
        if not string_protein_ids:
            continue
        if len(string_protein_ids) > 1:
            raise ValueError(
                f"Conflict row for {row[symbol_col]} contains multiple reviewed "
                f"STRING_Protein_ID values: {row[protein_id_col]}"
            )
        add_mapping(
            conflict_map,
            symbol=row[symbol_col].strip(),
            string_protein_id=string_protein_ids[0],
            source_name="conflicts input",
        )

    return conflict_map


def create_mapping_rows(
    hgnc_rows: list[dict[str, str]],
    symbol_col: str,
    index_col: str,
    direct_map: dict[str, str],
    missing_match_map: dict[str, str],
    conflict_map: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    output_rows = []
    counts = {
        "direct": 0,
        "missing_match": 0,
        "conflict": 0,
        "unmatched": 0,
    }

    for row in hgnc_rows:
        symbol = row[symbol_col].strip()
        string_protein_id = ""

        if symbol in direct_map:
            string_protein_id = direct_map[symbol]
            counts["direct"] += 1
        elif symbol in missing_match_map:
            string_protein_id = missing_match_map[symbol]
            counts["missing_match"] += 1
        elif symbol in conflict_map:
            string_protein_id = conflict_map[symbol]
            counts["conflict"] += 1
        else:
            counts["unmatched"] += 1

        output_rows.append(
            {
                "Symbol": symbol,
                "Index": row[index_col].strip(),
                "STRING_Protein_ID": string_protein_id,
            }
        )

    return output_rows, counts


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["Symbol", "Index", "STRING_Protein_ID"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        protein_input = validate_input(args.protein_input, "Protein input")
        missing_match_input = validate_input(args.missing_match_input, "Missing match input")
        conflicts_input = validate_input(args.conflicts_input, "Conflicts input")
        output = args.output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        protein_fieldnames, protein_rows = read_csv_rows(protein_input, delimiter="\t")
        missing_match_fieldnames, missing_match_rows = read_csv_rows(missing_match_input)
        conflicts_fieldnames, conflict_rows = read_csv_rows(conflicts_input)

        hgnc_symbol_col = find_column(hgnc_fieldnames, ("Symbol",), hgnc_input.name)
        hgnc_index_col = find_column(hgnc_fieldnames, ("Index",), hgnc_input.name)
        protein_id_col = find_column(
            protein_fieldnames,
            ("#string_protein_id", "string_protein_id"),
            protein_input.name,
        )
        preferred_name_col = find_column(
            protein_fieldnames,
            ("preferred_name",),
            protein_input.name,
        )
        missing_match_symbol_col = find_column(
            missing_match_fieldnames,
            ("Symbol", "symbol"),
            missing_match_input.name,
        )
        missing_match_protein_id_col = find_column(
            missing_match_fieldnames,
            ("STRING_Protein_ID", "string_protein_id"),
            missing_match_input.name,
        )
        conflict_symbol_col = find_column(
            conflicts_fieldnames,
            ("Symbol", "symbol"),
            conflicts_input.name,
        )
        conflict_protein_id_col = find_column(
            conflicts_fieldnames,
            ("STRING_Protein_ID", "string_protein_id"),
            conflicts_input.name,
        )

        direct_map = build_direct_map(
            protein_rows,
            protein_id_col=protein_id_col,
            preferred_name_col=preferred_name_col,
        )
        missing_match_map = build_missing_match_map(
            missing_match_rows,
            symbol_col=missing_match_symbol_col,
            protein_id_col=missing_match_protein_id_col,
        )
        conflict_map = build_conflict_map(
            conflict_rows,
            symbol_col=conflict_symbol_col,
            protein_id_col=conflict_protein_id_col,
        )
        output_rows, counts = create_mapping_rows(
            hgnc_rows,
            symbol_col=hgnc_symbol_col,
            index_col=hgnc_index_col,
            direct_map=direct_map,
            missing_match_map=missing_match_map,
            conflict_map=conflict_map,
        )

        write_output(output, output_rows)

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"Direct Symbol/preferred_name matches: {counts['direct']:,}")
        print(f"Rows filled from missing_match: {counts['missing_match']:,}")
        print(f"Rows filled from reviewed conflicts: {counts['conflict']:,}")
        print(f"Unmatched rows: {counts['unmatched']:,}")
        print(f"Output: {output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
