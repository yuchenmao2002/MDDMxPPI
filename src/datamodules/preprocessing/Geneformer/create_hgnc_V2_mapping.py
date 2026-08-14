#!/usr/bin/env python3
"""Create the final HGNC to Geneformer V2 Ensembl/token mapping."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
RAW_GENEFORMER_DIR = PROJECT_ROOT / "data" / "raw" / "Geneformer"
INTERIM_GENEFORMER_DIR = PROJECT_ROOT / "data" / "interim" / "Geneformer"
PROCESSED_GENEFORMER_DIR = PROJECT_ROOT / "data" / "processed" / "Geneformer"

DEFAULT_HGNC_INPUT = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"
DEFAULT_V2_INPUT = INTERIM_GENEFORMER_DIR / "V2_gene_name_id.csv"
DEFAULT_MISSING_MATCH_INPUT = INTERIM_GENEFORMER_DIR / "missing_match.csv"
DEFAULT_TOKEN_DICTIONARY_INPUT = (
    RAW_GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
)
DEFAULT_OUTPUT = PROCESSED_GENEFORMER_DIR / "hgnc_V2_mapping.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use hgnc_symbol.csv as the base vocabulary, add canonical Ensembl "
            "IDs from direct V2 gene_name_or_id matches and missing_match.csv, "
            "then add Geneformer V2 token IDs."
        )
    )
    parser.add_argument(
        "--hgnc-input",
        type=Path,
        default=DEFAULT_HGNC_INPUT,
        help=f"HGNC Symbol/Index CSV. Default: {DEFAULT_HGNC_INPUT}",
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=DEFAULT_V2_INPUT,
        help=f"Geneformer V2 gene-name mapping CSV. Default: {DEFAULT_V2_INPUT}",
    )
    parser.add_argument(
        "--missing-match-input",
        type=Path,
        default=DEFAULT_MISSING_MATCH_INPUT,
        help=(
            "Automatic mappings for HGNC symbols without direct V2 matches. "
            f"Default: {DEFAULT_MISSING_MATCH_INPUT}"
        ),
    )
    parser.add_argument(
        "--token-dictionary-input",
        type=Path,
        default=DEFAULT_TOKEN_DICTIONARY_INPUT,
        help=(
            "Geneformer V2 Ensembl-to-token pickle. "
            f"Default: {DEFAULT_TOKEN_DICTIONARY_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Final HGNC/Geneformer V2 mapping CSV. Default: {DEFAULT_OUTPUT}",
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


def validate_hgnc_rows(
    rows: list[dict[str, str]],
    symbol_col: str,
    index_col: str,
    source_name: str,
) -> set[str]:
    symbols: set[str] = set()
    indices: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        symbol = row[symbol_col].strip()
        index = row[index_col].strip()
        if not symbol:
            raise ValueError(f"{source_name} row {row_number} has an empty Symbol.")
        if not index:
            raise ValueError(f"{source_name} row {row_number} has an empty Index.")
        if symbol in symbols:
            raise ValueError(f"Duplicate HGNC Symbol in {source_name}: {symbol}")
        if index in indices:
            raise ValueError(f"Duplicate HGNC Index in {source_name}: {index}")
        symbols.add(symbol)
        indices.add(index)

    return symbols


def build_v2_map(
    rows: list[dict[str, str]],
    gene_name_col: str,
    canonical_ensembl_col: str,
    source_name: str,
) -> dict[str, str]:
    gene_name_to_ensembl: dict[str, str] = {}
    ensembl_to_gene_name: dict[str, str] = {}

    for row_number, row in enumerate(rows, start=2):
        gene_name_or_id = row[gene_name_col].strip()
        canonical_ensembl_id = row[canonical_ensembl_col].strip()
        if not gene_name_or_id or not canonical_ensembl_id:
            raise ValueError(
                f"{source_name} row {row_number} has an empty "
                "gene_name_or_id or canonical_ensembl_id."
            )

        existing_ensembl = gene_name_to_ensembl.get(gene_name_or_id)
        if existing_ensembl is not None:
            raise ValueError(
                f"Duplicate gene_name_or_id in {source_name}: {gene_name_or_id}"
            )
        existing_gene_name = ensembl_to_gene_name.get(canonical_ensembl_id)
        if existing_gene_name is not None:
            raise ValueError(
                f"Duplicate canonical_ensembl_id in {source_name}: "
                f"{canonical_ensembl_id} ({existing_gene_name}, {gene_name_or_id})"
            )

        gene_name_to_ensembl[gene_name_or_id] = canonical_ensembl_id
        ensembl_to_gene_name[canonical_ensembl_id] = gene_name_or_id

    return gene_name_to_ensembl


def build_missing_match_map(
    rows: list[dict[str, str]],
    symbol_col: str,
    matched_gene_name_col: str,
    canonical_ensembl_col: str,
    hgnc_symbols: set[str],
    direct_symbols: set[str],
    v2_map: dict[str, str],
    source_name: str,
) -> dict[str, str]:
    symbol_to_ensembl: dict[str, str] = {}
    ensembl_to_symbol: dict[str, str] = {}

    for row_number, row in enumerate(rows, start=2):
        symbol = row[symbol_col].strip()
        matched_gene_name = row[matched_gene_name_col].strip()
        canonical_ensembl_id = row[canonical_ensembl_col].strip()
        if not symbol or not matched_gene_name or not canonical_ensembl_id:
            raise ValueError(
                f"{source_name} row {row_number} contains an empty required value."
            )
        if symbol not in hgnc_symbols:
            raise ValueError(
                f"{source_name} contains Symbol absent from HGNC input: {symbol}"
            )
        if symbol in direct_symbols:
            raise ValueError(
                f"{source_name} contains directly matched HGNC Symbol: {symbol}"
            )
        if symbol in symbol_to_ensembl:
            raise ValueError(f"Duplicate Symbol in {source_name}: {symbol}")

        expected_ensembl_id = v2_map.get(matched_gene_name)
        if expected_ensembl_id is None:
            raise ValueError(
                f"{source_name} matched_gene_name_or_id is absent from V2 input: "
                f"{matched_gene_name}"
            )
        if expected_ensembl_id != canonical_ensembl_id:
            raise ValueError(
                f"Inconsistent V2 mapping for {symbol}: {matched_gene_name} maps "
                f"to {expected_ensembl_id}, not {canonical_ensembl_id}"
            )

        existing_symbol = ensembl_to_symbol.get(canonical_ensembl_id)
        if existing_symbol is not None:
            raise ValueError(
                f"Duplicate supplemental canonical_ensembl_id "
                f"{canonical_ensembl_id}: {existing_symbol}, {symbol}"
            )
        symbol_to_ensembl[symbol] = canonical_ensembl_id
        ensembl_to_symbol[canonical_ensembl_id] = symbol

    return symbol_to_ensembl


def load_token_dictionary(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        payload = pickle.load(source)

    if not isinstance(payload, dict):
        raise TypeError(f"Token dictionary must be a dict, got {type(payload).__name__}.")

    token_dictionary: dict[str, str] = {}
    token_id_to_key: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Token dictionary contains a non-string key: {key!r}"
            )
        try:
            token_id = str(int(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Token dictionary contains a non-integer token ID for {key}: "
                f"{value!r}"
            ) from error
        if int(token_id) < 0:
            raise ValueError(f"Token dictionary contains a negative token ID: {key}")

        existing_key = token_id_to_key.get(token_id)
        if existing_key is not None:
            raise ValueError(
                f"Duplicate token ID {token_id}: {existing_key}, {key}"
            )
        token_dictionary[key] = token_id
        token_id_to_key[token_id] = key

    return token_dictionary


def create_mapping_rows(
    hgnc_rows: list[dict[str, str]],
    symbol_col: str,
    index_col: str,
    v2_map: dict[str, str],
    missing_match_map: dict[str, str],
    token_dictionary: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    output_rows: list[dict[str, str]] = []
    counts = {
        "direct": 0,
        "missing_match": 0,
        "unmatched": 0,
        "token_found": 0,
    }
    ensembl_to_symbol: dict[str, str] = {}
    token_id_to_symbol: dict[str, str] = {}
    missing_token_ids: set[str] = set()

    for row in hgnc_rows:
        symbol = row[symbol_col].strip()
        ensembl_id = ""

        if symbol in v2_map:
            ensembl_id = v2_map[symbol]
            counts["direct"] += 1
        elif symbol in missing_match_map:
            ensembl_id = missing_match_map[symbol]
            counts["missing_match"] += 1
        else:
            counts["unmatched"] += 1

        token_id = ""
        if ensembl_id:
            existing_symbol = ensembl_to_symbol.get(ensembl_id)
            if existing_symbol is not None:
                raise ValueError(
                    f"Final mapping assigns Ensembl ID {ensembl_id} to both "
                    f"{existing_symbol} and {symbol}."
                )
            ensembl_to_symbol[ensembl_id] = symbol

            token_id = token_dictionary.get(ensembl_id, "")
            if not token_id:
                missing_token_ids.add(ensembl_id)
            else:
                existing_token_symbol = token_id_to_symbol.get(token_id)
                if existing_token_symbol is not None:
                    raise ValueError(
                        f"Final mapping assigns Token_ID {token_id} to both "
                        f"{existing_token_symbol} and {symbol}."
                    )
                token_id_to_symbol[token_id] = symbol
                counts["token_found"] += 1

        output_rows.append(
            {
                "Symbol": symbol,
                "Index": row[index_col].strip(),
                "Ensembl_ID": ensembl_id,
                "Token_ID": token_id,
            }
        )

    if missing_token_ids:
        examples = ", ".join(sorted(missing_token_ids)[:10])
        raise ValueError(
            f"{len(missing_token_ids):,} mapped Ensembl IDs are absent from the "
            f"token dictionary. Examples: {examples}"
        )

    return output_rows, counts


def write_mapping_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["Symbol", "Index", "Ensembl_ID", "Token_ID"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        hgnc_input = validate_input(args.hgnc_input, "HGNC input")
        v2_input = validate_input(args.v2_input, "Geneformer V2 input")
        missing_match_input = validate_input(
            args.missing_match_input,
            "Missing match input",
        )
        token_dictionary_input = validate_input(
            args.token_dictionary_input,
            "Token dictionary input",
        )
        output = args.output.expanduser().absolute()

        hgnc_fieldnames, hgnc_rows = read_csv_rows(hgnc_input)
        v2_fieldnames, v2_rows = read_csv_rows(v2_input)
        missing_fieldnames, missing_rows = read_csv_rows(missing_match_input)

        hgnc_symbol_col = find_column(
            hgnc_fieldnames,
            ("Symbol", "symbol"),
            hgnc_input.name,
        )
        hgnc_index_col = find_column(
            hgnc_fieldnames,
            ("Index", "index"),
            hgnc_input.name,
        )
        v2_gene_name_col = find_column(
            v2_fieldnames,
            ("gene_name_or_id",),
            v2_input.name,
        )
        v2_ensembl_col = find_column(
            v2_fieldnames,
            ("canonical_ensembl_id",),
            v2_input.name,
        )
        missing_symbol_col = find_column(
            missing_fieldnames,
            ("Symbol", "symbol"),
            missing_match_input.name,
        )
        missing_gene_name_col = find_column(
            missing_fieldnames,
            ("matched_gene_name_or_id",),
            missing_match_input.name,
        )
        missing_ensembl_col = find_column(
            missing_fieldnames,
            ("canonical_ensembl_id",),
            missing_match_input.name,
        )

        hgnc_symbols = validate_hgnc_rows(
            hgnc_rows,
            symbol_col=hgnc_symbol_col,
            index_col=hgnc_index_col,
            source_name=hgnc_input.name,
        )
        v2_map = build_v2_map(
            v2_rows,
            gene_name_col=v2_gene_name_col,
            canonical_ensembl_col=v2_ensembl_col,
            source_name=v2_input.name,
        )
        direct_symbols = hgnc_symbols.intersection(v2_map)
        missing_match_map = build_missing_match_map(
            missing_rows,
            symbol_col=missing_symbol_col,
            matched_gene_name_col=missing_gene_name_col,
            canonical_ensembl_col=missing_ensembl_col,
            hgnc_symbols=hgnc_symbols,
            direct_symbols=direct_symbols,
            v2_map=v2_map,
            source_name=missing_match_input.name,
        )
        token_dictionary = load_token_dictionary(token_dictionary_input)

        v2_ensembl_ids = set(v2_map.values())
        v2_ids_missing_tokens = sorted(v2_ensembl_ids - token_dictionary.keys())
        if v2_ids_missing_tokens:
            examples = ", ".join(v2_ids_missing_tokens[:10])
            raise ValueError(
                f"{len(v2_ids_missing_tokens):,} V2 canonical Ensembl IDs are "
                f"absent from the token dictionary. Examples: {examples}"
            )

        output_rows, counts = create_mapping_rows(
            hgnc_rows,
            symbol_col=hgnc_symbol_col,
            index_col=hgnc_index_col,
            v2_map=v2_map,
            missing_match_map=missing_match_map,
            token_dictionary=token_dictionary,
        )
        mapped_rows = counts["direct"] + counts["missing_match"]
        if counts["token_found"] != mapped_rows:
            raise ValueError(
                f"Token coverage check failed: found {counts['token_found']:,} "
                f"for {mapped_rows:,} mapped Ensembl IDs."
            )

        write_mapping_csv(output, output_rows)

        print(f"HGNC rows: {len(hgnc_rows):,}")
        print(f"Direct Symbol/gene_name_or_id matches: {counts['direct']:,}")
        print(f"Rows filled from missing_match: {counts['missing_match']:,}")
        print(f"Mapped Ensembl IDs: {mapped_rows:,}")
        print(f"Unmatched rows: {counts['unmatched']:,}")
        print(
            f"Token IDs found: {counts['token_found']:,}/{mapped_rows:,} "
            "mapped Ensembl IDs"
        )
        print(
            f"V2 Ensembl IDs found in token dictionary: "
            f"{len(v2_ensembl_ids):,}/{len(v2_ensembl_ids):,}"
        )
        print(f"Output: {output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
