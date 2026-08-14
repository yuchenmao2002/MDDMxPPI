#!/usr/bin/env python3
"""Classify Geneformer V2 matches using queried HGNC information."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_GENEFORMER_DIR = PROJECT_ROOT / "data" / "interim" / "Geneformer"

DEFAULT_HGNC_INPUT = INTERIM_GENEFORMER_DIR / "hgnc_missing_with_info.csv"
DEFAULT_V2_INPUT = INTERIM_GENEFORMER_DIR / "V2_missing.csv"
DEFAULT_MATCH_OUTPUT = INTERIM_GENEFORMER_DIR / "missing_match.csv"
DEFAULT_CONFLICTS_OUTPUT = INTERIM_GENEFORMER_DIR / "missing_match_conflicts.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "First match HGNC Ensembl IDs against V2_missing "
            "canonical_ensembl_id values. For rows without an Ensembl match, "
            "fall back to matching Previous and Alias values against "
            "gene_name_or_id. Unique Ensembl/Previous matches are written to "
            "a match file; Alias matches and multi-hit rows are written to a "
            "conflicts file for manual review."
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
        "--v2-input",
        type=Path,
        default=DEFAULT_V2_INPUT,
        help=f"Geneformer V2 missing-gene CSV. Default: {DEFAULT_V2_INPUT}",
    )
    parser.add_argument(
        "--match-output",
        type=Path,
        default=DEFAULT_MATCH_OUTPUT,
        help=(
            "Output CSV for unique Ensembl/Previous matches. "
            f"Default: {DEFAULT_MATCH_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default=DEFAULT_CONFLICTS_OUTPUT,
        help=(
            "Output CSV for Alias matches and rows with multiple possible "
            f"matches. Default: {DEFAULT_CONFLICTS_OUTPUT}"
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


def build_v2_maps(
    rows: list[dict[str, str]],
    gene_name_col: str,
    canonical_ensembl_col: str,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    gene_name_to_canonical_ensembl: dict[str, str] = {}
    canonical_ensembl_to_gene_names: dict[str, list[str]] = {}
    duplicate_gene_names = []

    for row in rows:
        gene_name_or_id = row[gene_name_col].strip()
        canonical_ensembl_id = row[canonical_ensembl_col].strip()
        if not gene_name_or_id:
            continue
        if gene_name_or_id in gene_name_to_canonical_ensembl:
            duplicate_gene_names.append(gene_name_or_id)
        gene_name_to_canonical_ensembl[gene_name_or_id] = canonical_ensembl_id
        if canonical_ensembl_id:
            canonical_ensembl_to_gene_names.setdefault(
                canonical_ensembl_id,
                [],
            ).append(gene_name_or_id)

    if duplicate_gene_names:
        examples = ", ".join(sorted(set(duplicate_gene_names))[:10])
        raise ValueError(f"Duplicate gene_name_or_id values in V2 input: {examples}")

    return gene_name_to_canonical_ensembl, canonical_ensembl_to_gene_names


def match_gene_name_values(
    row: dict[str, str],
    column: str,
    gene_name_to_canonical_ensembl: dict[str, str],
) -> list[str]:
    return [
        value
        for value in split_values(row.get(column, ""))
        if value in gene_name_to_canonical_ensembl
    ]


def match_ensembl_values(
    row: dict[str, str],
    column: str,
    canonical_ensembl_to_gene_names: dict[str, list[str]],
) -> list[str]:
    matches = []
    for ensembl_id in split_values(row.get(column, "")):
        matches.extend(canonical_ensembl_to_gene_names.get(ensembl_id, []))
    return unique_in_order(matches)


def build_conflict_row(
    symbol: str,
    gene_name_matches: list[str],
    gene_name_to_canonical_ensembl: dict[str, str],
) -> dict[str, str]:
    return {
        "Symbol": symbol,
        "potential_gene_name_or_id": "; ".join(gene_name_matches),
        "potential_canonical_ensembl_id": "; ".join(
            gene_name_to_canonical_ensembl[gene_name_or_id]
            for gene_name_or_id in gene_name_matches
        ),
        "canonical_ensembl_id": "",
    }


def match_row(
    row: dict[str, str],
    symbol_col: str,
    previous_col: str,
    alias_col: str,
    ensembl_col: str,
    gene_name_to_canonical_ensembl: dict[str, str],
    canonical_ensembl_to_gene_names: dict[str, list[str]],
) -> tuple[dict[str, str] | None, dict[str, str] | None, str]:
    symbol = row[symbol_col].strip()

    ensembl_matches = match_ensembl_values(
        row,
        ensembl_col,
        canonical_ensembl_to_gene_names,
    )
    if len(ensembl_matches) == 1:
        gene_name_or_id = ensembl_matches[0]
        return {
            "Symbol": symbol,
            "matched_gene_name_or_id": gene_name_or_id,
            "canonical_ensembl_id": gene_name_to_canonical_ensembl[gene_name_or_id],
        }, None, "Ensembl"
    if len(ensembl_matches) > 1:
        return None, build_conflict_row(
            symbol,
            ensembl_matches,
            gene_name_to_canonical_ensembl,
        ), "Multiple"

    previous_matches = match_gene_name_values(
        row,
        previous_col,
        gene_name_to_canonical_ensembl,
    )
    alias_matches = match_gene_name_values(
        row,
        alias_col,
        gene_name_to_canonical_ensembl,
    )
    previous_matches = unique_in_order(previous_matches)
    all_matches = unique_in_order(previous_matches + alias_matches)

    if not all_matches:
        return None, None, "Unmatched"

    if alias_matches or len(all_matches) > 1:
        if alias_matches and len(all_matches) > 1:
            reason = "Alias and Multiple"
        elif alias_matches:
            reason = "Alias"
        else:
            reason = "Multiple"
        return None, build_conflict_row(
            symbol,
            all_matches,
            gene_name_to_canonical_ensembl,
        ), reason

    gene_name_or_id = previous_matches[0]
    return {
        "Symbol": symbol,
        "matched_gene_name_or_id": gene_name_or_id,
        "canonical_ensembl_id": gene_name_to_canonical_ensembl[gene_name_or_id],
    }, None, "Previous"


def main() -> int:
    args = parse_args()

    try:
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        v2_input = validate_input(args.v2_input, "Geneformer V2 input")
        match_output = args.match_output.expanduser().absolute()
        conflicts_output = args.conflicts_output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        v2_fieldnames, v2_rows = read_csv_rows(v2_input)

        symbol_col = find_column(hgnc_fieldnames, ("Symbol", "symbol"), hgnc_input.name)
        previous_col = find_column(
            hgnc_fieldnames,
            ("Previous", "previous"),
            hgnc_input.name,
        )
        alias_col = find_column(hgnc_fieldnames, ("Alias", "alias"), hgnc_input.name)
        ensembl_col = find_column(
            hgnc_fieldnames,
            ("Ensembl", "ensembl"),
            hgnc_input.name,
        )
        gene_name_col = find_column(
            v2_fieldnames,
            ("gene_name_or_id",),
            v2_input.name,
        )
        canonical_ensembl_col = find_column(
            v2_fieldnames,
            ("canonical_ensembl_id",),
            v2_input.name,
        )

        (
            gene_name_to_canonical_ensembl,
            canonical_ensembl_to_gene_names,
        ) = build_v2_maps(
            v2_rows,
            gene_name_col=gene_name_col,
            canonical_ensembl_col=canonical_ensembl_col,
        )
        match_rows: list[dict[str, str]] = []
        conflict_rows: list[dict[str, str]] = []
        unmatched_rows = 0
        ensembl_match_rows = 0
        previous_match_rows = 0
        alias_review_rows = 0
        multiple_review_rows = 0

        for row in hgnc_rows:
            match_row_output, conflict_row, classification = match_row(
                row,
                symbol_col=symbol_col,
                previous_col=previous_col,
                alias_col=alias_col,
                ensembl_col=ensembl_col,
                gene_name_to_canonical_ensembl=gene_name_to_canonical_ensembl,
                canonical_ensembl_to_gene_names=canonical_ensembl_to_gene_names,
            )
            if match_row_output is not None:
                match_rows.append(match_row_output)
                if classification == "Ensembl":
                    ensembl_match_rows += 1
                elif classification == "Previous":
                    previous_match_rows += 1
            elif conflict_row is not None:
                conflict_rows.append(conflict_row)
                if "Alias" in classification:
                    alias_review_rows += 1
                if "Multiple" in classification:
                    multiple_review_rows += 1
            else:
                unmatched_rows += 1

        write_csv_rows(
            match_output,
            ["Symbol", "matched_gene_name_or_id", "canonical_ensembl_id"],
            match_rows,
        )
        write_csv_rows(
            conflicts_output,
            [
                "Symbol",
                "potential_gene_name_or_id",
                "potential_canonical_ensembl_id",
                "canonical_ensembl_id",
            ],
            conflict_rows,
        )

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(
            "Geneformer V2 missing gene_name_or_id values: "
            f"{len(gene_name_to_canonical_ensembl):,}"
        )
        print(f"Unique Ensembl matches: {ensembl_match_rows:,}")
        print(f"Unique Previous fallback matches: {previous_match_rows:,}")
        print(f"Alias matches for manual review: {alias_review_rows:,}")
        print(f"Multi-hit rows for manual review: {multiple_review_rows:,}")
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
