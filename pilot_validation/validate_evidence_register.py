#!/usr/bin/env python3
"""Validate evidence-register completeness for the Voice AI pilot package."""

from __future__ import annotations

import argparse
import csv
from datetime import date
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
    "artifact",
    "source_url",
    "confidence",
    "caveat",
    "last_checked",
    "next_review",
    "owner",
]

ALLOWED_CONFIDENCE_LEVELS = ("High", "Medium", "Low")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pilot_validation/evidence_register.csv completeness.")
    parser.add_argument(
        "path",
        nargs="?",
        default="pilot_validation/evidence_register.csv",
        help="Evidence register CSV path.",
    )
    return parser.parse_args()


def parse_iso_date(value: str, row_number: int, column: str, errors: list[str]) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        errors.append(f"row {row_number}: {column} must be ISO date YYYY-MM-DD")
        return None


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

        source_url = (row.get("source_url") or "").strip()
        if source_url and not source_url.startswith(("http://", "https://")):
            errors.append(f"row {row_number}: source_url must start with http:// or https://")

        confidence = (row.get("confidence") or "").strip()
        if confidence and not confidence.startswith(ALLOWED_CONFIDENCE_LEVELS):
            errors.append(
                f"row {row_number}: confidence must start with one of "
                f"{', '.join(ALLOWED_CONFIDENCE_LEVELS)}"
            )

        last_checked = (row.get("last_checked") or "").strip()
        next_review = (row.get("next_review") or "").strip()
        if last_checked and next_review:
            last_checked_date = parse_iso_date(last_checked, row_number, "last_checked", errors)
            next_review_date = parse_iso_date(next_review, row_number, "next_review", errors)
            if last_checked_date and next_review_date and next_review_date <= last_checked_date:
                errors.append(f"row {row_number}: next_review must be after last_checked")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Evidence register validation passed: {len(rows)} rows checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
