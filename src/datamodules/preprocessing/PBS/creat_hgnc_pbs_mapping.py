#!/usr/bin/env python3
"""Create the final HGNC symbol to PBS index mapping."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"
PROCESSED_PBS_DIR = PROJECT_ROOT / "data" / "processed" / "PBS"

DEFAULT_HGNC_INPUT = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"
DEFAULT_HGNC_MISSING_INPUT = INTERIM_PBS_DIR / "hgnc_missing.csv"
DEFAULT_MISSING_MAPPING_INPUT = INTERIM_PBS_DIR / "missing_mapping.csv"
DEFAULT_CONFLICTS_INPUT = INTERIM_PBS_DIR / "missing_mapping_conflicts.csv"
DEFAULT_OUTPUT = PROCESSED_PBS_DIR / "hgnc_pbs_mapping.csv"
DEFAULT_REPORT = INTERIM_PBS_DIR / "hgnc_pbs_mapping_report.txt"
DEFAULT_PLACEHOLDER = "__NO_PBS_INDEX__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create hgnc_pbs_mapping.csv from the full HGNC vocabulary, direct "
            "HGNC/PBS matches, automatic missing mappings, manually reviewed "
            "conflicts, and unresolved HGNC missing symbols."
        )
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=f"HGNC symbol vocabulary CSV. Default: {DEFAULT_HGNC_INPUT}",
    )
    parser.add_argument(
        "--hgnc-missing-input",
        type=Path,
        default=DEFAULT_HGNC_MISSING_INPUT,
        help=f"HGNC rows without direct PBS matches. Default: {DEFAULT_HGNC_MISSING_INPUT}",
    )
    parser.add_argument(
        "--missing-mapping-input",
        type=Path,
        default=DEFAULT_MISSING_MAPPING_INPUT,
        help=f"Automatic missing mapping CSV. Default: {DEFAULT_MISSING_MAPPING_INPUT}",
    )
    parser.add_argument(
        "--conflicts-input",
        type=Path,
        default=DEFAULT_CONFLICTS_INPUT,
        help=f"Manually reviewed conflict mapping CSV. Default: {DEFAULT_CONFLICTS_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output mapping CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Output summary report TXT. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--placeholder",
        default=DEFAULT_PLACEHOLDER,
        help=f"PBS_Index placeholder for unresolved symbols. Default: {DEFAULT_PLACEHOLDER}",
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


def validate_input(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def split_values(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def build_symbol_set(
    rows: list[dict[str, str]],
    symbol_col: str,
    source_name: str,
) -> set[str]:
    symbols: set[str] = set()
    duplicate_symbols = set()

    for row in rows:
        symbol = row[symbol_col].strip()
        if not symbol:
            raise ValueError(f"{source_name} contains an empty Symbol value.")
        if symbol in symbols:
            duplicate_symbols.add(symbol)
        symbols.add(symbol)

    if duplicate_symbols:
        examples = ", ".join(sorted(duplicate_symbols)[:10])
        raise ValueError(f"Duplicate Symbol values in {source_name}: {examples}")

    return symbols


def add_mapping_rows(
    symbol_to_pbs_index: dict[str, str],
    rows: list[dict[str, str]],
    symbol_col: str,
    pbs_index_col: str,
    source_name: str,
    skip_empty_pbs_index: bool,
) -> int:
    added = 0

    for row in rows:
        symbol = row[symbol_col].strip()
        pbs_index = row[pbs_index_col].strip()
        if not symbol:
            raise ValueError(f"{source_name} contains an empty Symbol value.")
        if not pbs_index and skip_empty_pbs_index:
            continue
        if not pbs_index:
            raise ValueError(f"{source_name} has an empty PBS_Index for Symbol={symbol!r}.")

        existing = symbol_to_pbs_index.get(symbol)
        if existing is not None and existing != pbs_index:
            raise ValueError(
                f"Conflicting PBS_Index values for Symbol={symbol!r}: "
                f"{existing!r} vs {pbs_index!r} from {source_name}"
            )

        if existing is None:
            added += 1
        symbol_to_pbs_index[symbol] = pbs_index

    return added


def create_mapping_rows(
    hgnc_rows: list[dict[str, str]],
    symbol_col: str,
    index_col: str,
    hgnc_missing_symbols: set[str],
    symbol_to_pbs_index: dict[str, str],
    placeholder: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    mapping_rows: list[dict[str, str]] = []
    counts = {
        "direct": 0,
        "mapped_missing": 0,
        "placeholder": 0,
    }

    for row in hgnc_rows:
        symbol = row[symbol_col].strip()
        if symbol in symbol_to_pbs_index:
            pbs_index = symbol_to_pbs_index[symbol]
            counts["mapped_missing"] += 1
        elif symbol in hgnc_missing_symbols:
            pbs_index = placeholder
            counts["placeholder"] += 1
        else:
            pbs_index = symbol
            counts["direct"] += 1

        mapping_rows.append(
            {
                "Symbol": symbol,
                "Index": row[index_col],
                "PBS_Index": pbs_index,
            }
        )

    return mapping_rows, counts


def write_mapping_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["Symbol", "Index", "PBS_Index"])
        writer.writeheader()
        writer.writerows(rows)


def is_multi_pbs_index(value: str, placeholder: str) -> bool:
    return value != placeholder and len(split_values(value)) > 1


def write_report(
    path: Path,
    mapping_rows: list[dict[str, str]],
    hgnc_input: Path,
    hgnc_missing_input: Path,
    missing_mapping_input: Path,
    conflicts_input: Path,
    output: Path,
    placeholder: str,
    counts: dict[str, int],
    automatic_mapping_count: int,
    reviewed_conflict_mapping_count: int,
) -> None:
    multi_pbs_index_rows = [
        row for row in mapping_rows if is_multi_pbs_index(row["PBS_Index"], placeholder)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        target.write("HGNC symbol to PBS index mapping report\n")
        target.write(f"HGNC input: {hgnc_input}\n")
        target.write(f"HGNC missing input: {hgnc_missing_input}\n")
        target.write(f"Missing mapping input: {missing_mapping_input}\n")
        target.write(f"Conflicts input: {conflicts_input}\n")
        target.write(f"Mapping output: {output}\n")
        target.write(f"Placeholder: {placeholder}\n\n")
        target.write(f"Total genes: {len(mapping_rows):,}\n")
        target.write(f"Direct Symbol == PBS_Index rows: {counts['direct']:,}\n")
        target.write(f"Mapped missing rows: {counts['mapped_missing']:,}\n")
        target.write(f"  From missing_mapping: {automatic_mapping_count:,}\n")
        target.write(f"  From reviewed conflicts: {reviewed_conflict_mapping_count:,}\n")
        target.write(f"Placeholder rows: {counts['placeholder']:,}\n")
        target.write(
            f"Rows with multiple PBS_Index values: {len(multi_pbs_index_rows):,}\n\n"
        )

        target.write("Rows with multiple PBS_Index values\n")
        target.write("Symbol,Index,PBS_Index\n")
        if multi_pbs_index_rows:
            for row in multi_pbs_index_rows:
                target.write(f"{row['Symbol']},{row['Index']},{row['PBS_Index']}\n")
        else:
            target.write("(none)\n")


def main() -> int:
    args = parse_args()

    try:
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        hgnc_missing_input = validate_input(args.hgnc_missing_input, "HGNC missing input")
        missing_mapping_input = validate_input(
            args.missing_mapping_input,
            "Missing mapping input",
        )
        conflicts_input = validate_input(args.conflicts_input, "Conflicts input")
        output = args.output.expanduser().absolute()
        report = args.report.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        hgnc_missing_fieldnames, hgnc_missing_rows = read_csv_rows(hgnc_missing_input)
        missing_mapping_fieldnames, missing_mapping_rows = read_csv_rows(
            missing_mapping_input
        )
        conflicts_fieldnames, conflict_rows = read_csv_rows(conflicts_input)

        hgnc_symbol_col = find_column(hgnc_fieldnames, ("Symbol", "symbol"), hgnc_input.name)
        hgnc_index_col = find_column(hgnc_fieldnames, ("Index", "index"), hgnc_input.name)
        hgnc_missing_symbol_col = find_column(
            hgnc_missing_fieldnames,
            ("Symbol", "symbol"),
            hgnc_missing_input.name,
        )
        missing_mapping_symbol_col = find_column(
            missing_mapping_fieldnames,
            ("Symbol", "symbol"),
            missing_mapping_input.name,
        )
        missing_mapping_pbs_index_col = find_column(
            missing_mapping_fieldnames,
            ("PBS_Index", "pbs_index"),
            missing_mapping_input.name,
        )
        conflicts_symbol_col = find_column(
            conflicts_fieldnames,
            ("Symbol", "symbol"),
            conflicts_input.name,
        )
        conflicts_pbs_index_col = find_column(
            conflicts_fieldnames,
            ("PBS_Index", "pbs_index"),
            conflicts_input.name,
        )

        hgnc_missing_symbols = build_symbol_set(
            hgnc_missing_rows,
            symbol_col=hgnc_missing_symbol_col,
            source_name=hgnc_missing_input.name,
        )

        symbol_to_pbs_index: dict[str, str] = {}
        automatic_mapping_count = add_mapping_rows(
            symbol_to_pbs_index,
            missing_mapping_rows,
            symbol_col=missing_mapping_symbol_col,
            pbs_index_col=missing_mapping_pbs_index_col,
            source_name=missing_mapping_input.name,
            skip_empty_pbs_index=False,
        )
        reviewed_conflict_mapping_count = add_mapping_rows(
            symbol_to_pbs_index,
            conflict_rows,
            symbol_col=conflicts_symbol_col,
            pbs_index_col=conflicts_pbs_index_col,
            source_name=conflicts_input.name,
            skip_empty_pbs_index=True,
        )

        unknown_symbols = sorted(set(symbol_to_pbs_index) - hgnc_missing_symbols)
        if unknown_symbols:
            examples = ", ".join(unknown_symbols[:10])
            raise ValueError(
                "Missing mapping inputs contain symbols absent from hgnc_missing.csv: "
                f"{examples}"
            )

        mapping_rows, counts = create_mapping_rows(
            hgnc_rows,
            symbol_col=hgnc_symbol_col,
            index_col=hgnc_index_col,
            hgnc_missing_symbols=hgnc_missing_symbols,
            symbol_to_pbs_index=symbol_to_pbs_index,
            placeholder=args.placeholder,
        )

        write_mapping_csv(output, mapping_rows)
        write_report(
            report,
            mapping_rows=mapping_rows,
            hgnc_input=hgnc_input,
            hgnc_missing_input=hgnc_missing_input,
            missing_mapping_input=missing_mapping_input,
            conflicts_input=conflicts_input,
            output=output,
            placeholder=args.placeholder,
            counts=counts,
            automatic_mapping_count=automatic_mapping_count,
            reviewed_conflict_mapping_count=reviewed_conflict_mapping_count,
        )

        print(f"Mapping rows: {len(mapping_rows):,}")
        print(f"Direct Symbol == PBS_Index rows: {counts['direct']:,}")
        print(f"Mapped missing rows: {counts['mapped_missing']:,}")
        print(f"  From missing_mapping: {automatic_mapping_count:,}")
        print(f"  From reviewed conflicts: {reviewed_conflict_mapping_count:,}")
        print(f"Placeholder rows: {counts['placeholder']:,}")
        print(f"Output: {output}")
        print(f"Report: {report}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
