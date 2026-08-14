#!/usr/bin/env python3
"""Keep STRING PPI rows whose endpoints both have HGNC mappings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).absolute().parents[4]
RAW_STRING_DIR = PROJECT_ROOT / "data" / "raw" / "STRING"
INTERIM_STRING_DIR = PROJECT_ROOT / "data" / "interim" / "STRING"
PROCESSED_STRING_DIR = PROJECT_ROOT / "data" / "processed" / "STRING"

DEFAULT_MAPPING_INPUT = PROCESSED_STRING_DIR / "hgnc_string_mapping.csv"
DEFAULT_PPI_INPUT = RAW_STRING_DIR / "9606.protein.links.full.v12.0.onlyAB.csv"
DEFAULT_OUTPUT = INTERIM_STRING_DIR / "PPI_string_protein_id.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read mapped STRING protein IDs from hgnc_string_mapping.csv and "
            "remove any PPI rows whose protein1 or protein2 is absent from that "
            "mapping."
        )
    )
    parser.add_argument(
        "--mapping-input",
        type=Path,
        default=DEFAULT_MAPPING_INPUT,
        help=f"HGNC to STRING mapping CSV. Default: {DEFAULT_MAPPING_INPUT}",
    )
    parser.add_argument(
        "--ppi-input",
        type=Path,
        default=DEFAULT_PPI_INPUT,
        help=f"STRING PPI CSV. Default: {DEFAULT_PPI_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Filtered PPI output CSV. Default: {DEFAULT_OUTPUT}",
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


def split_values(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def collect_mapped_protein_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")

        string_protein_id_col = find_column(
            reader.fieldnames,
            ("STRING_Protein_ID", "string_protein_id"),
            path.name,
        )

        mapped_ids = set()
        for row in reader:
            mapped_ids.update(split_values(row.get(string_protein_id_col, "")))

    if not mapped_ids:
        raise ValueError(f"No mapped STRING protein IDs found in {path}")

    return mapped_ids


def filter_ppi_rows(
    ppi_input: Path,
    output: Path,
    mapped_ids: set[str],
) -> tuple[int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)

    with ppi_input.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{ppi_input} does not contain a CSV header.")

        protein1_col = find_column(reader.fieldnames, ("protein1",), ppi_input.name)
        protein2_col = find_column(reader.fieldnames, ("protein2",), ppi_input.name)

        with output.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()

            total_rows = 0
            kept_rows = 0
            removed_rows = 0

            for row in reader:
                total_rows += 1
                if (
                    row[protein1_col].strip() in mapped_ids
                    and row[protein2_col].strip() in mapped_ids
                ):
                    writer.writerow(row)
                    kept_rows += 1
                else:
                    removed_rows += 1

    return total_rows, kept_rows, removed_rows


def main() -> int:
    args = parse_args()

    try:
        mapping_input = validate_input(args.mapping_input, "Mapping input")
        ppi_input = validate_input(args.ppi_input, "PPI input")
        output = args.output.expanduser().absolute()

        mapped_ids = collect_mapped_protein_ids(mapping_input)
        total_rows, kept_rows, removed_rows = filter_ppi_rows(
            ppi_input,
            output,
            mapped_ids=mapped_ids,
        )

        print(f"Mapped STRING protein IDs: {len(mapped_ids):,}")
        print(f"PPI rows: {total_rows:,}")
        print(f"Removed PPI rows: {removed_rows:,}")
        print(f"Kept PPI rows: {kept_rows:,}")
        print(f"Output: {output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
