#!/usr/bin/env python3
"""Create a compact HGNC gene-symbol CSV with a zero-based index."""

from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).absolute().parents[3]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "PBS" / "hgnc_symbol.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "hgnc_symbol.csv"

OUTPUT_COLUMNS = ["Symbol", "Index", "Name", "ID", "URL"]


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")

        with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for index, row in enumerate(reader):
                writer.writerow(
                    {
                        "Symbol": row["Symbol"],
                        "Index": index,
                        "Name": row["Name"],
                        "ID": row["ID"],
                        "URL": row["URL"],
                    }
                )


if __name__ == "__main__":
    main()
