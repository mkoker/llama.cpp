#!/usr/bin/env python3
"""Validate evidence-register completeness for the Voice AI pilot package."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED_COLUMNS = [
    "assumption",
    "artifact",
    "source_url",
    "confidence",
    "caveat",
    "last_checked",
    "next_review",
    "owner",
]
REQUIRED_NON_BLANK = [
    "assumption",
    "source_url",
    "confidence",
    "caveat",
    "last_checked",
    "next_review",
    "owner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pilot_validation/evidence_register.csv completeness.")
    parser.add_argument(
        "path",
        nargs="?",
        default="pilot_validation/evidence_register.csv",
        help="Evidence register CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    errors: list[str] = []

    if not path.exists():
        print(f"ERROR: missing evidence register: {path}")
        return 1

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            errors.append(f"missing required columns: {', '.join(missing_columns)}")
        rows = list(reader)

    if not rows:
        errors.append("evidence register must contain at least one evidence row")

    for row_number, row in enumerate(rows, start=2):
        for column in REQUIRED_NON_BLANK:
            if not (row.get(column) or "").strip():
                errors.append(f"row {row_number}: {column} is blank")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Evidence register validation passed: {len(rows)} rows checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
