"""CSV and JSON report exporters for benchmark results."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

BASE_CSV_FIELDS = [
    "agent",
    "task_id",
    "task_name",
    "model",
    "success",
    "latency_seconds",
    "usage.prompt_tokens",
    "usage.completion_tokens",
    "usage.total_tokens",
    "error",
    "output",
]


def export_results(results: list[Any], output_path: str | Path, summary: dict[str, Any] | None = None) -> Path:
    """Export benchmark results to JSON or CSV and return the written path."""

    path = Path(output_path)
    normalized_results = [_normalize_result(result) for result in results]
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _export_json(normalized_results, path, summary=summary)
    if suffix == ".csv":
        return _export_csv(normalized_results, path)

    raise ValueError("unsupported report format: expected .json or .csv")


def _export_json(results: list[dict[str, Any]], path: Path, *, summary: dict[str, Any] | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary or {}, "results": results}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _export_csv(results: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_result(result) for result in results]
    fieldnames = _csv_fieldnames(rows)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "to_dict"):
        value = result.to_dict()
        if isinstance(value, dict):
            return value
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    raise TypeError(f"unsupported result type: {type(result).__name__}")


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in result.items():
        if key == "usage" and isinstance(value, dict):
            for usage_key, usage_value in value.items():
                flattened[f"usage.{usage_key}"] = usage_value
        elif key == "metadata" and isinstance(value, dict):
            for metadata_key, metadata_value in value.items():
                flattened[f"metadata.{metadata_key}"] = metadata_value
        else:
            flattened[key] = value
    return flattened


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames = list(BASE_CSV_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


__all__ = ["export_results"]
