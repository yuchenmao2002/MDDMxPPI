#!/usr/bin/env python3
"""Add HGNC previous, alias, and Ensembl IDs to missing-symbol rows."""

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
INTERIM_PBS_DIR = PROJECT_ROOT / "data" / "interim" / "PBS"

DEFAULT_INPUT = INTERIM_PBS_DIR / "hgnc_missing.csv"
DEFAULT_OUTPUT = INTERIM_PBS_DIR / "hgnc_missing_with_info.csv"

DROP_COLUMNS = {"index", "name", "id", "url"}
HGNC_ID_PATTERN = re.compile(r"HGNC:\d+")
REST_URL_TEMPLATE = "https://rest.genenames.org/fetch/hgnc_id/{query}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop unused HGNC columns and add Previous/Alias symbols plus "
            "Ensembl IDs from HGNC."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input missing-symbol CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV with Previous, Alias, and Ensembl columns. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="HGNC IDs to request at once. Default: 100",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20,
        help="HTTP timeout in seconds for each HGNC request. Default: 20",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per HGNC request after the first attempt. Default: 3",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1,
        help="Seconds to wait between retries. Default: 1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N input rows. Useful for testing.",
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


def extract_hgnc_id(url: str) -> str:
    match = HGNC_ID_PATTERN.search(url)
    if match is None:
        raise ValueError(f"Could not find an HGNC ID in URL: {url}")
    return match.group(0)


def as_symbol_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return [str(value)]


def join_symbols(value) -> str:
    return "; ".join(as_symbol_list(value))


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def fetch_hgnc_symbols_batch(
    hgnc_ids: list[str],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, tuple[str, str, str]]:
    query = " OR ".join(hgnc_ids)
    rest_url = REST_URL_TEMPLATE.format(query=quote(query, safe=":"))
    request = Request(
        rest_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "CodeNo0-hgnc-symbol-annotation/1.0",
        },
    )

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            docs = payload.get("response", {}).get("docs", [])
            return {
                doc["hgnc_id"]: (
                    join_symbols(doc.get("prev_symbol")),
                    join_symbols(doc.get("alias_symbol")),
                    join_symbols(doc.get("ensembl_gene_id")),
                )
                for doc in docs
                if "hgnc_id" in doc
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt >= retries:
                first_id = hgnc_ids[0] if hgnc_ids else "<empty>"
                last_id = hgnc_ids[-1] if hgnc_ids else "<empty>"
                raise RuntimeError(
                    f"Failed to fetch HGNC batch {first_id}..{last_id}: {exc}"
                ) from exc
            time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError("Failed to fetch HGNC batch.")


def annotate_rows(
    rows: list[dict[str, str]],
    url_column: str,
    timeout: float,
    retries: int,
    retry_delay: float,
    batch_size: int,
) -> list[tuple[str, str, str]]:
    hgnc_ids = [extract_hgnc_id(row[url_column]) for row in rows]
    unique_hgnc_ids = list(dict.fromkeys(hgnc_ids))
    id_to_symbols: dict[str, tuple[str, str, str]] = {}
    id_batches = chunked(unique_hgnc_ids, max(1, batch_size))

    for batch_index, hgnc_id_batch in enumerate(id_batches, start=1):
        id_to_symbols.update(
            fetch_hgnc_symbols_batch(
                hgnc_id_batch,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
            )
        )
        print(
            f"Fetched HGNC batches: {batch_index:,}/{len(id_batches):,} "
            f"({min(batch_index * batch_size, len(unique_hgnc_ids)):,}/"
            f"{len(unique_hgnc_ids):,} IDs)",
            flush=True,
        )

    return [id_to_symbols.get(hgnc_id, ("", "", "")) for hgnc_id in hgnc_ids]


def validate_input(path: Path) -> Path:
    path = path.expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")
    return path


def main() -> int:
    args = parse_args()

    try:
        input_path = validate_input(args.input)
        output_path = args.output.expanduser().absolute()

        fieldnames, rows = read_csv_rows(input_path)
        if args.limit is not None:
            rows = rows[: args.limit]

        url_column = find_column(fieldnames, ("URL", "url"), input_path.name)
        find_column(fieldnames, ("Symbol", "symbol"), input_path.name)

        output_fieldnames = [
            fieldname for fieldname in fieldnames if fieldname.lower() not in DROP_COLUMNS
        ]
        for added_column in ("Previous", "Alias", "Ensembl"):
            if added_column not in output_fieldnames:
                output_fieldnames.append(added_column)

        annotations = annotate_rows(
            rows,
            url_column=url_column,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
            batch_size=args.batch_size,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=output_fieldnames)
            writer.writeheader()

            for row, (previous_symbols, alias_symbols, ensembl_id) in zip(
                rows, annotations
            ):
                output_row = {
                    fieldname: value
                    for fieldname, value in row.items()
                    if fieldname in output_fieldnames
                }
                output_row["Previous"] = previous_symbols
                output_row["Alias"] = alias_symbols
                output_row["Ensembl"] = ensembl_id
                writer.writerow(output_row)

        print(f"Input rows: {len(rows):,}")
        print(f"Output columns: {', '.join(output_fieldnames)}")
        print(f"Output: {output_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
