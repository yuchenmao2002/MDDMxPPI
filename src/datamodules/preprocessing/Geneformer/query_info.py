#!/usr/bin/env python3
"""Add HGNC previous, alias, and Ensembl IDs to Geneformer-missing HGNC rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INTERIM_GENEFORMER_DIR = PROJECT_ROOT / "data" / "interim" / "Geneformer"

DEFAULT_INPUT = INTERIM_GENEFORMER_DIR / "hgnc_missing.csv"
DEFAULT_OUTPUT = INTERIM_GENEFORMER_DIR / "hgnc_missing_with_info.csv"

DROP_COLUMNS = {"index", "name", "id", "url"}
HGNC_ID_PATTERN = re.compile(r"HGNC:\d+")
REST_URL_TEMPLATE = "https://rest.genenames.org/fetch/hgnc_id/{query}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query HGNC for previous symbols, alias symbols, and Ensembl IDs "
            "for rows in Geneformer/hgnc_missing.csv."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of HGNC IDs queried in each request (default: 100).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20,
        help="Timeout in seconds for each request (default: 20).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request after the first attempt (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1,
        help="Initial delay in seconds between retries (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows; useful for testing.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        rows = list(reader)

    return list(reader.fieldnames), rows


def extract_hgnc_id(row: dict[str, str], row_number: int) -> str:
    match = HGNC_ID_PATTERN.search(row.get("URL", ""))
    if match is None:
        raise ValueError(
            f"Could not find an HGNC ID in URL for input row {row_number}: "
            f"{row.get('URL', '')!r}"
        )
    return match.group(0)


def fetch_batch(
    hgnc_ids: list[str],
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> list[dict[str, object]]:
    query = quote(" OR ".join(hgnc_ids), safe=":")
    url = REST_URL_TEMPLATE.format(query=query)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "CodeNo0-hgnc-geneformer-missing-annotation/1.0",
        },
    )

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            return payload["response"]["docs"]
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
            if attempt >= retries:
                raise RuntimeError(
                    f"HGNC query failed after {retries + 1} attempts for "
                    f"{hgnc_ids[0]} through {hgnc_ids[-1]}"
                ) from error
            time.sleep(retry_delay * (attempt + 1))

    raise AssertionError("Unreachable")


def format_hgnc_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def query_annotations(
    hgnc_ids: list[str],
    *,
    batch_size: int,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, dict[str, str]]:
    annotations: dict[str, dict[str, str]] = {}
    unique_hgnc_ids = list(dict.fromkeys(hgnc_ids))
    total_batches = (len(unique_hgnc_ids) + batch_size - 1) // batch_size

    for start in range(0, len(unique_hgnc_ids), batch_size):
        batch = unique_hgnc_ids[start : start + batch_size]
        batch_number = start // batch_size + 1
        print(
            f"Querying batch {batch_number}/{total_batches} "
            f"({len(batch)} HGNC IDs)...",
            flush=True,
        )
        documents = fetch_batch(
            batch,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )

        for document in documents:
            hgnc_id = document.get("hgnc_id")
            if not isinstance(hgnc_id, str):
                continue
            annotations[hgnc_id] = {
                "Previous": format_hgnc_field(document.get("prev_symbol")),
                "Alias": format_hgnc_field(document.get("alias_symbol")),
                "Ensembl": format_hgnc_field(document.get("ensembl_gene_id")),
            }

    return annotations


def write_rows(
    path: Path,
    input_fieldnames: list[str],
    rows: list[dict[str, str]],
    hgnc_ids: list[str],
    annotations: dict[str, dict[str, str]],
) -> list[str]:
    retained_fields = [
        field for field in input_fieldnames if field.lower() not in DROP_COLUMNS
    ]
    output_fieldnames = retained_fields + ["Previous", "Alias", "Ensembl"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        for row, hgnc_id in zip(rows, hgnc_ids):
            output_row = {field: row.get(field, "") for field in retained_fields}
            output_row.update(
                annotations.get(
                    hgnc_id,
                    {"Previous": "", "Alias": "", "Ensembl": ""},
                )
            )
            writer.writerow(output_row)

    return output_fieldnames


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay cannot be negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit cannot be negative")

    input_fieldnames, rows = read_rows(args.input)
    if "URL" not in input_fieldnames:
        raise ValueError("Input CSV must contain a URL column with HGNC IDs")
    if args.limit is not None:
        rows = rows[: args.limit]

    hgnc_ids = [
        extract_hgnc_id(row, row_number)
        for row_number, row in enumerate(rows, start=2)
    ]
    annotations = query_annotations(
        hgnc_ids,
        batch_size=args.batch_size,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    output_fieldnames = write_rows(
        args.output,
        input_fieldnames,
        rows,
        hgnc_ids,
        annotations,
    )

    missing_responses = sum(hgnc_id not in annotations for hgnc_id in hgnc_ids)
    print(f"Input rows: {len(rows)}")
    print(f"HGNC records returned: {len(annotations)}")
    print(f"Rows without an HGNC response: {missing_responses}")
    print(f"Output columns: {', '.join(output_fieldnames)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
