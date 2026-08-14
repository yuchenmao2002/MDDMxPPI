#!/usr/bin/env python3
"""Convert STRING protein IDs in the PPI table to HGNC vocabulary indices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).absolute().parents[4]
INTERIM_STRING_DIR = PROJECT_ROOT / "data" / "interim" / "STRING"
PROCESSED_STRING_DIR = PROJECT_ROOT / "data" / "processed" / "STRING"

DEFAULT_PPI_INPUT = INTERIM_STRING_DIR / "PPI_string_protein_id.csv"
DEFAULT_MAPPING_INPUT = PROCESSED_STRING_DIR / "hgnc_string_mapping.csv"
DEFAULT_OUTPUT = PROCESSED_STRING_DIR / "PPI_index.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read PPI_string_protein_id.csv and replace protein1/protein2 STRING "
            "protein IDs with Index1/Index2 from hgnc_string_mapping.csv."
        )
    )
    parser.add_argument(
        "--ppi-input",
        type=Path,
        default=DEFAULT_PPI_INPUT,
        help=f"Filtered STRING PPI CSV. Default: {DEFAULT_PPI_INPUT}",
    )
    parser.add_argument(
        "--mapping-input",
        type=Path,
        default=DEFAULT_MAPPING_INPUT,
        help=f"Symbol to STRING protein ID mapping CSV. Default: {DEFAULT_MAPPING_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PPI index CSV. Default: {DEFAULT_OUTPUT}",
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


def build_string_id_to_index(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")

        index_col = find_column(reader.fieldnames, ("Index", "index"), path.name)
        string_id_col = find_column(
            reader.fieldnames,
            ("STRING_Protein_ID", "string_protein_id"),
            path.name,
        )

        string_id_to_index: dict[str, str] = {}
        duplicate_conflicts: list[str] = []

        for row in reader:
            index = row[index_col].strip()
            for string_protein_id in split_values(row.get(string_id_col, "")):
                previous_index = string_id_to_index.get(string_protein_id)
                if previous_index is not None and previous_index != index:
                    duplicate_conflicts.append(
                        f"{string_protein_id}: {previous_index} vs {index}"
                    )
                string_id_to_index[string_protein_id] = index

    if duplicate_conflicts:
        examples = "; ".join(duplicate_conflicts[:10])
        raise ValueError(
            "The mapping contains STRING_Protein_ID values assigned to multiple "
            f"indices. Examples: {examples}"
        )

    return string_id_to_index


def convert_ppi_to_indices(
    ppi_input: Path,
    output: Path,
    string_id_to_index: dict[str, str],
) -> tuple[int, int]:
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp")

    total_rows = 0
    missing_ids: dict[str, int] = {}

    try:
        with ppi_input.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"{ppi_input} does not contain a CSV header.")

            protein1_col = find_column(reader.fieldnames, ("protein1",), ppi_input.name)
            protein2_col = find_column(reader.fieldnames, ("protein2",), ppi_input.name)
            trailing_fieldnames = [
                fieldname
                for fieldname in reader.fieldnames
                if fieldname not in (protein1_col, protein2_col)
            ]
            output_fieldnames = ["Index1", "Index2"] + trailing_fieldnames

            with tmp_output.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=output_fieldnames)
                writer.writeheader()

                for row in reader:
                    total_rows += 1
                    protein1 = row[protein1_col].strip()
                    protein2 = row[protein2_col].strip()
                    index1 = string_id_to_index.get(protein1)
                    index2 = string_id_to_index.get(protein2)

                    if index1 is None:
                        missing_ids[protein1] = missing_ids.get(protein1, 0) + 1
                    if index2 is None:
                        missing_ids[protein2] = missing_ids.get(protein2, 0) + 1
                    if index1 is None or index2 is None:
                        continue

                    output_row = {
                        "Index1": index1,
                        "Index2": index2,
                    }
                    output_row.update({fieldname: row[fieldname] for fieldname in trailing_fieldnames})
                    writer.writerow(output_row)

        if missing_ids:
            examples = "; ".join(
                f"{protein_id} ({count:,} rows)"
                for protein_id, count in list(missing_ids.items())[:20]
            )
            raise ValueError(
                f"{len(missing_ids):,} STRING protein IDs in PPI input have no "
                f"Index mapping. Examples: {examples}"
            )

        tmp_output.replace(output)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    return total_rows, len(string_id_to_index)


def main() -> int:
    args = parse_args()

    try:
        ppi_input = validate_input(args.ppi_input, "PPI input")
        mapping_input = validate_input(args.mapping_input, "Mapping input")
        output = args.output.expanduser().absolute()

        string_id_to_index = build_string_id_to_index(mapping_input)
        ppi_rows, mapped_string_ids = convert_ppi_to_indices(
            ppi_input,
            output,
            string_id_to_index=string_id_to_index,
        )

        print(f"Mapped STRING protein IDs: {mapped_string_ids:,}")
        print(f"PPI rows converted: {ppi_rows:,}")
        print(f"Output: {output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
