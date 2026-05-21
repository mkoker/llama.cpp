#!/usr/bin/env python3
"""Voice AI pilot benchmark harness.

Dry-run mode generates deterministic synthetic measurements so the validation
package can be tested without provider credentials, paid API calls, audio files,
or PHI/regulated data exposure.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Scenario:
    vertical: str
    workflow: str
    provider_stack: str
    target_p95_ms: int
    synthetic_mean_ms: int
    synthetic_jitter_ms: int
    synthetic_error_rate: float
    synthetic_fallback_rate: float
    notes: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        vertical="Healthcare Omni",
        workflow="ambient_documentation_turn",
        provider_stack="native_realtime_omni",
        target_p95_ms=1200,
        synthetic_mean_ms=620,
        synthetic_jitter_ms=180,
        synthetic_error_rate=0.015,
        synthetic_fallback_rate=0.005,
        notes="Synthetic non-PHI timing sample for clinical turn-taking validation.",
    ),
    Scenario(
        vertical="Contact Center Cascade",
        workflow="inbound_routing_and_summary",
        provider_stack="streaming_stt_llm_tts_cascade",
        target_p95_ms=1800,
        synthetic_mean_ms=940,
        synthetic_jitter_ms=260,
        synthetic_error_rate=0.025,
        synthetic_fallback_rate=0.020,
        notes="Synthetic cascade timing sample with STT/LLM/TTS component timings.",
    ),
    Scenario(
        vertical="Fraud Operations",
        workflow="voice_verification_alert_triage",
        provider_stack="realtime_voice_fraud_assist",
        target_p95_ms=1500,
        synthetic_mean_ms=780,
        synthetic_jitter_ms=230,
        synthetic_error_rate=0.020,
        synthetic_fallback_rate=0.015,
        notes="Synthetic regulated-workflow timing sample; no biometric data used.",
    ),
)


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


def bounded_gaussian(rng: random.Random, mean_ms: int, jitter_ms: int) -> int:
    value = rng.gauss(mean_ms, jitter_ms)
    return max(80, int(round(value)))


def component_timings(total_ms: int, scenario: Scenario, rng: random.Random) -> dict[str, int]:
    if "cascade" in scenario.provider_stack:
        endpointing = max(40, int(total_ms * rng.uniform(0.18, 0.28)))
        stt = max(25, int(total_ms * rng.uniform(0.10, 0.18)))
        llm = max(30, int(total_ms * rng.uniform(0.22, 0.34)))
        tts = max(35, total_ms - endpointing - stt - llm)
        return {
            "endpointing_ms": endpointing,
            "stt_ms": stt,
            "llm_ms": llm,
            "tts_first_audio_ms": tts,
        }

    endpointing = max(40, int(total_ms * rng.uniform(0.22, 0.34)))
    model_audio = max(35, total_ms - endpointing)
    return {
        "endpointing_ms": endpointing,
        "native_model_first_audio_ms": model_audio,
    }


def synthesize_samples(scenario: Scenario, iterations: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for idx in range(iterations):
        latency_ms = bounded_gaussian(rng, scenario.synthetic_mean_ms, scenario.synthetic_jitter_ms)
        error = rng.random() < scenario.synthetic_error_rate
        fallback = (not error) and rng.random() < scenario.synthetic_fallback_rate
        samples.append(
            {
                "sample_id": idx + 1,
                "latency_ms": latency_ms,
                "component_timings": component_timings(latency_ms, scenario, rng),
                "success": not error,
                "provider_error": error,
                "fallback_used": fallback,
            }
        )
    return samples


def summarize_samples(samples: Iterable[dict[str, Any]], target_p95_ms: int) -> dict[str, Any]:
    sample_list = list(samples)
    latencies = [float(sample["latency_ms"]) for sample in sample_list if sample.get("success")]
    errors = sum(1 for sample in sample_list if sample.get("provider_error"))
    fallbacks = sum(1 for sample in sample_list if sample.get("fallback_used"))
    attempts = len(sample_list)
    successes = len(latencies)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    return {
        "attempts": attempts,
        "successes": successes,
        "success_rate": round(successes / attempts, 4) if attempts else 0.0,
        "error_count": errors,
        "fallback_count": fallbacks,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "target_p95": target_p95_ms,
            "p95_pass": p95 <= target_p95_ms,
        },
    }


def run_dry_run(iterations: int, seed: int) -> dict[str, Any]:
    scenario_results = []
    for offset, scenario in enumerate(SCENARIOS):
        samples = synthesize_samples(scenario, iterations, seed + offset)
        scenario_results.append(
            {
                "vertical": scenario.vertical,
                "workflow": scenario.workflow,
                "provider_stack": scenario.provider_stack,
                "mode": "dry_run",
                "data_classification": "synthetic_no_phi_no_biometrics_no_customer_data",
                "notes": scenario.notes,
                "summary": summarize_samples(samples, scenario.target_p95_ms),
                "samples": samples,
            }
        )

    return {
        "generated_at": "dry-run-deterministic",
        "harness_version": "1.0",
        "dry_run": True,
        "external_api_calls": 0,
        "iterations_per_scenario": iterations,
        "seed": seed,
        "scenarios": scenario_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Voice AI pilot benchmark harness. Use --dry-run for synthetic offline output."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic synthetic benchmark output without provider credentials or API calls.",
    )
    parser.add_argument(
        "--out",
        default="pilot_validation/dry_run_results.json",
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=24,
        help="Synthetic samples per scenario in dry-run mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260521,
        help="Random seed for deterministic dry-run output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit(
            "Live provider benchmarking is not implemented yet. Run with --dry-run to validate output schema without API keys."
        )
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    result = run_dry_run(iterations=args.iterations, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} with {len(result['scenarios'])} dry-run scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
