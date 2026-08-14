#!/usr/bin/env python3
"""Create the Geneformer V2-316M gene-name-to-Ensembl mapping CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import re
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
RAW_GENEFORMER_DIR = PROJECT_ROOT / "data" / "raw" / "Geneformer"
INTERIM_GENEFORMER_DIR = PROJECT_ROOT / "data" / "interim" / "Geneformer"

DEFAULT_GENE_NAME_ID_INPUT = RAW_GENEFORMER_DIR / "gene_name_id_dict_gc104M.pkl"
DEFAULT_TOKEN_DICTIONARY_INPUT = RAW_GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
DEFAULT_OUTPUT = INTERIM_GENEFORMER_DIR / "gene_name_id_mapping_v2_316m.csv"

OUTPUT_COLUMNS = ["gene_name_or_id", "canonical_ensembl_id"]
CANONICAL_ENSEMBL_ID_PATTERN = re.compile(r"^ENSG\d{11}$")
EXPECTED_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<mask>": 1,
    "<cls>": 2,
    "<eos>": 3,
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickle plain built-in containers without importing global objects."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"Loading global objects is not allowed: {module}.{name}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter gene_name_id_dict_gc104M.pkl to mappings whose "
            "canonical_ensembl_id is present in the Geneformer V2-316M "
            "token vocabulary."
        )
    )
    parser.add_argument(
        "--gene-name-id-input",
        type=Path,
        default=DEFAULT_GENE_NAME_ID_INPUT,
        help=(
            "Input gene-name/ID to canonical Ensembl ID pickle. "
            f"Default: {DEFAULT_GENE_NAME_ID_INPUT}"
        ),
    )
    parser.add_argument(
        "--token-dictionary-input",
        type=Path,
        default=DEFAULT_TOKEN_DICTIONARY_INPUT,
        help=(
            "Input Geneformer V2 token dictionary pickle. "
            f"Default: {DEFAULT_TOKEN_DICTIONARY_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output filtered mapping CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def validate_input(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def load_plain_dictionary(path: Path, label: str) -> dict:
    try:
        with path.open("rb") as source:
            obj = RestrictedUnpickler(source).load()
    except pickle.UnpicklingError as exc:
        raise ValueError(f"Unable to safely load {label}: {path}: {exc}") from exc

    if not isinstance(obj, dict):
        raise TypeError(
            f"{label} must contain a top-level dict, found {type(obj).__name__}: {path}"
        )
    return obj


def get_v2_gene_ids(token_dictionary: dict) -> set[str]:
    if not all(
        isinstance(key, str) and type(token_id) is int
        for key, token_id in token_dictionary.items()
    ):
        raise TypeError("Token dictionary must contain only str -> int mappings.")

    special_tokens = {
        key: token_id
        for key, token_id in token_dictionary.items()
        if key.startswith("<")
    }
    if special_tokens != EXPECTED_SPECIAL_TOKENS:
        raise ValueError(
            "Unexpected V2 special tokens: "
            f"expected {EXPECTED_SPECIAL_TOKENS}, found {special_tokens}"
        )

    token_ids = list(token_dictionary.values())
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("Token dictionary contains duplicate token IDs.")
    if set(token_ids) != set(range(len(token_dictionary))):
        raise ValueError("Token dictionary IDs are not contiguous from 0 to vocab_size - 1.")

    v2_gene_ids = set(token_dictionary) - set(special_tokens)
    invalid_gene_ids = sorted(
        gene_id
        for gene_id in v2_gene_ids
        if CANONICAL_ENSEMBL_ID_PATTERN.fullmatch(gene_id) is None
    )
    if invalid_gene_ids:
        examples = ", ".join(invalid_gene_ids[:10])
        raise ValueError(f"Token dictionary contains invalid Ensembl gene IDs: {examples}")

    return v2_gene_ids


def create_filtered_rows(
    gene_name_id_dictionary: dict,
    v2_gene_ids: set[str],
) -> list[dict[str, str]]:
    if not all(
        isinstance(gene_name_or_id, str) and isinstance(canonical_ensembl_id, str)
        for gene_name_or_id, canonical_ensembl_id in gene_name_id_dictionary.items()
    ):
        raise TypeError(
            "Gene name/ID dictionary must contain only str -> str mappings."
        )

    invalid_targets = sorted(
        canonical_ensembl_id
        for canonical_ensembl_id in gene_name_id_dictionary.values()
        if CANONICAL_ENSEMBL_ID_PATTERN.fullmatch(canonical_ensembl_id) is None
    )
    if invalid_targets:
        examples = ", ".join(invalid_targets[:10])
        raise ValueError(
            "Gene name/ID dictionary contains invalid canonical Ensembl IDs: "
            f"{examples}"
        )

    rows = [
        {
            "gene_name_or_id": gene_name_or_id,
            "canonical_ensembl_id": canonical_ensembl_id,
        }
        for gene_name_or_id, canonical_ensembl_id in gene_name_id_dictionary.items()
        if canonical_ensembl_id in v2_gene_ids
    ]
    rows.sort(
        key=lambda row: (
            row["gene_name_or_id"],
            row["canonical_ensembl_id"],
        )
    )

    source_values = [row["gene_name_or_id"] for row in rows]
    canonical_values = [row["canonical_ensembl_id"] for row in rows]

    if len(source_values) != len(set(source_values)):
        raise ValueError("Filtered mappings contain duplicate gene_name_or_id values.")
    if len(canonical_values) != len(set(canonical_values)):
        raise ValueError("Filtered mappings contain duplicate canonical_ensembl_id values.")

    retained_gene_ids = set(canonical_values)
    if retained_gene_ids != v2_gene_ids:
        missing_gene_ids = sorted(v2_gene_ids - retained_gene_ids)
        examples = ", ".join(missing_gene_ids[:10])
        raise ValueError(
            f"Gene name/ID dictionary does not cover {len(missing_gene_ids):,} "
            f"V2 genes. Examples: {examples}"
        )

    return rows


def write_output(path: Path, rows: list[dict[str, str]], overwrite: bool) -> Path:
    path = path.expanduser().absolute()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def main() -> int:
    args = parse_args()

    try:
        gene_name_id_input = validate_input(
            args.gene_name_id_input,
            "Gene name/ID dictionary input",
        )
        token_dictionary_input = validate_input(
            args.token_dictionary_input,
            "Token dictionary input",
        )

        gene_name_id_dictionary = load_plain_dictionary(
            gene_name_id_input,
            "gene name/ID dictionary",
        )
        token_dictionary = load_plain_dictionary(
            token_dictionary_input,
            "token dictionary",
        )

        v2_gene_ids = get_v2_gene_ids(token_dictionary)
        rows = create_filtered_rows(gene_name_id_dictionary, v2_gene_ids)
        output = write_output(args.output, rows, overwrite=args.overwrite)
    except (EOFError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Source mappings: {len(gene_name_id_dictionary):,}")
    print(f"V2 gene vocabulary: {len(v2_gene_ids):,}")
    print(f"Mappings retained: {len(rows):,}")
    print(f"Mappings removed: {len(gene_name_id_dictionary) - len(rows):,}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
