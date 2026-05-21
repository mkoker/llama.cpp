#!/usr/bin/env python3
"""Stdlib validation checks for Voice AI pilot benchmark results.

This intentionally does not depend on jsonschema so the dry-run gate can execute
on a clean VM with only Python installed. result_schema.json remains the formal
contract; this script enforces the operational checks that matter for pilot
readiness and catches schema drift in benchmark_harness.py output.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_VERTICALS = {
    "Healthcare Omni",
    "Contact Center Cascade",
    "Fraud Operations",
}
SYNTHETIC_CLASSIFICATION = "synthetic_no_phi_no_biometrics_no_customer_data"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_schema_document(schema: dict[str, Any], errors: list[str]) -> None:
    require(schema.get("type") == "object", "schema root must be an object", errors)
    required = set(schema.get("required", []))
    for key in {
        "generated_at",
        "harness_version",
        "dry_run",
        "external_api_calls",
        "iterations_per_scenario",
        "seed",
        "scenarios",
    }:
        require(key in required, f"schema missing required top-level key: {key}", errors)
    require("scenario_result" in schema.get("$defs", {}), "schema missing scenario_result definition", errors)
    require("sample" in schema.get("$defs", {}), "schema missing sample definition", errors)


def validate_result(result: dict[str, Any], errors: list[str]) -> None:
    for key in [
        "generated_at",
        "harness_version",
        "dry_run",
        "external_api_calls",
        "iterations_per_scenario",
        "seed",
        "scenarios",
    ]:
        require(key in result, f"result missing top-level key: {key}", errors)

    require(isinstance(result.get("dry_run"), bool), "dry_run must be boolean", errors)
    require(isinstance(result.get("external_api_calls"), int), "external_api_calls must be integer", errors)
    if result.get("dry_run") is True:
        require(result.get("external_api_calls") == 0, "dry-run results must not record external API calls", errors)

    iterations = result.get("iterations_per_scenario")
    require(isinstance(iterations, int) and iterations >= 1, "iterations_per_scenario must be >= 1", errors)

    scenarios = result.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) >= 1, "scenarios must be a non-empty list", errors)
    if not isinstance(scenarios, list):
        return

    seen_verticals: set[str] = set()
    for idx, scenario in enumerate(scenarios, start=1):
        prefix = f"scenario[{idx}]"
        if not isinstance(scenario, dict):
            errors.append(f"{prefix} must be object")
            continue

        vertical = scenario.get("vertical")
        seen_verticals.add(str(vertical))
        require(vertical in EXPECTED_VERTICALS, f"{prefix}.vertical is not expected: {vertical!r}", errors)
        require(scenario.get("mode") in {"dry_run", "live"}, f"{prefix}.mode must be dry_run or live", errors)
        if result.get("dry_run") is True:
            require(scenario.get("mode") == "dry_run", f"{prefix}.mode must be dry_run for dry-run result", errors)
            require(
                scenario.get("data_classification") == SYNTHETIC_CLASSIFICATION,
                f"{prefix}.data_classification must prove no PHI/biometrics/customer data in dry-run",
                errors,
            )

        samples = scenario.get("samples")
        summary = scenario.get("summary")
        require(isinstance(samples, list) and len(samples) >= 1, f"{prefix}.samples must be non-empty list", errors)
        require(isinstance(summary, dict), f"{prefix}.summary must be object", errors)
        if not isinstance(samples, list) or not isinstance(summary, dict):
            continue

        if isinstance(iterations, int):
            require(len(samples) == iterations, f"{prefix}.samples length must equal iterations_per_scenario", errors)
        sample_ids = [sample.get("sample_id") for sample in samples if isinstance(sample, dict)]
        require(sample_ids == list(range(1, len(samples) + 1)), f"{prefix}.sample_id values must be sequential from 1", errors)

        successes = [sample for sample in samples if isinstance(sample, dict) and sample.get("success") is True]
        errors_count = sum(1 for sample in samples if isinstance(sample, dict) and sample.get("provider_error") is True)
        fallback_count = sum(1 for sample in samples if isinstance(sample, dict) and sample.get("fallback_used") is True)
        success_latencies = [float(sample["latency_ms"]) for sample in successes if isinstance(sample.get("latency_ms"), int)]

        for sample in samples:
            if not isinstance(sample, dict):
                errors.append(f"{prefix}.sample must be object")
                continue
            sid = sample.get("sample_id", "?")
            require(isinstance(sample.get("latency_ms"), int) and sample["latency_ms"] >= 0, f"{prefix}.sample[{sid}].latency_ms must be non-negative integer", errors)
            require(isinstance(sample.get("component_timings"), dict) and sample["component_timings"], f"{prefix}.sample[{sid}].component_timings must be non-empty object", errors)
            require(isinstance(sample.get("success"), bool), f"{prefix}.sample[{sid}].success must be boolean", errors)
            require(isinstance(sample.get("provider_error"), bool), f"{prefix}.sample[{sid}].provider_error must be boolean", errors)
            require(isinstance(sample.get("fallback_used"), bool), f"{prefix}.sample[{sid}].fallback_used must be boolean", errors)
            if sample.get("provider_error") is True:
                require(sample.get("success") is False, f"{prefix}.sample[{sid}] provider_error implies success=false", errors)
            if sample.get("fallback_used") is True:
                require(sample.get("success") is True, f"{prefix}.sample[{sid}] fallback_used implies success=true", errors)

        require(summary.get("attempts") == len(samples), f"{prefix}.summary.attempts must equal sample count", errors)
        require(summary.get("successes") == len(successes), f"{prefix}.summary.successes must equal success sample count", errors)
        require(summary.get("error_count") == errors_count, f"{prefix}.summary.error_count mismatch", errors)
        require(summary.get("fallback_count") == fallback_count, f"{prefix}.summary.fallback_count mismatch", errors)

        expected_success_rate = round(len(successes) / len(samples), 4) if samples else 0.0
        require(summary.get("success_rate") == expected_success_rate, f"{prefix}.summary.success_rate mismatch", errors)

        latency = summary.get("latency_ms", {})
        require(isinstance(latency, dict), f"{prefix}.summary.latency_ms must be object", errors)
        if isinstance(latency, dict):
            expected_mean = round(statistics.mean(success_latencies), 2) if success_latencies else 0.0
            expected_p50 = round(percentile(success_latencies, 50), 2)
            expected_p95 = round(percentile(success_latencies, 95), 2)
            expected_p99 = round(percentile(success_latencies, 99), 2)
            require(latency.get("mean") == expected_mean, f"{prefix}.latency.mean mismatch", errors)
            require(latency.get("p50") == expected_p50, f"{prefix}.latency.p50 mismatch", errors)
            require(latency.get("p95") == expected_p95, f"{prefix}.latency.p95 mismatch", errors)
            require(latency.get("p99") == expected_p99, f"{prefix}.latency.p99 mismatch", errors)
            target = latency.get("target_p95")
            require(isinstance(target, int) and target > 0, f"{prefix}.latency.target_p95 must be positive integer", errors)
            if isinstance(target, int):
                require(latency.get("p95_pass") == (expected_p95 <= target), f"{prefix}.latency.p95_pass mismatch", errors)

    require(seen_verticals == EXPECTED_VERTICALS, f"result must include exactly expected verticals: {sorted(EXPECTED_VERTICALS)}", errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Voice AI pilot benchmark schema and result JSON.")
    parser.add_argument("--schema", default="pilot_validation/result_schema.json", help="Path to formal JSON schema document.")
    parser.add_argument("--result", default="pilot_validation/dry_run_results.json", help="Path to benchmark result JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    schema = load_json(Path(args.schema))
    result = load_json(Path(args.result))
    validate_schema_document(schema, errors)
    validate_result(result, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Validation passed: {args.result} conforms to pilot validation checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
