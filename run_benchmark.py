#!/usr/bin/env python3
"""Main benchmark runner for the agent SDK comparison harness.

Runs selected SDK adapters across selected benchmark tasks for one or more
iterations, collecting latency, token usage, and validation success/failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from metrics.collector import MetricsCollector
from reports.exporter import export_results
from sdks.anthropic_agent import AnthropicAgent
from sdks.gemma_agent import GemmaAgent
from sdks.goose_agent import GooseAgent
from sdks.openai_agent import OpenAIAgent
from tasks.basic_tasks import BenchmarkTask, all_tasks, get_task, task_ids

AGENT_FACTORIES = {
    "openai": OpenAIAgent,
    "anthropic": AnthropicAgent,
    "goose": GooseAgent,
    "gemma": GemmaAgent,
}
DEFAULT_AGENT_NAMES = ("openai", "anthropic", "goose", "gemma")


class BenchmarkMockAgent:
    """Offline-safe deterministic agent for harness validation."""

    name = "mock"
    model = "benchmark-mock"

    def run(self, prompt: str, **_: Any) -> dict[str, Any]:
        output = _mock_output_for_prompt(prompt)
        prompt_count = len(prompt.split())
        completion_count = len(output.split())
        return {
            "agent": self.name,
            "model": self.model,
            "output": output,
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": completion_count,
                "total_tokens": prompt_count + completion_count,
            },
        }


def _mock_output_for_prompt(prompt: str) -> str:
    """Return deterministic output that satisfies the built-in task validators."""

    if "two recent AI model" in prompt:
        return (
            "- Example AI announcement, ExampleOrg, 2026-05-01, https://example.com/ai-1\n"
            "- Example agent SDK announcement, SDKOrg, 2026-05-02, https://example.com/ai-2"
        )
    if "slugify" in prompt:
        return """import re

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

assert slugify('Hello, World!') == 'hello-world'
assert slugify(' A  B ') == 'a-b'
assert slugify('x_y') == 'x-y'
""".strip()
    if "events.jsonl" in prompt:
        return """import json
from collections import defaultdict

totals = defaultdict(int)
with open('input/events.jsonl') as f:
    for line in f:
        row = json.loads(line)
        totals[row['team']] += row['score']
with open('output/team_scores.json', 'w') as f:
    json.dump(dict(totals), f, indent=2)
""".strip()
    if "INV-1042" in prompt:
        return json.dumps(
            {
                "invoice_id": "INV-1042",
                "vendor": "Northwind Tools",
                "invoice_date": "2026-05-01",
                "due_date": "2026-05-31",
                "line_items": [],
                "subtotal": 73.50,
                "tax": 4.41,
                "total": 77.91,
            }
        )
    if "A=2h" in prompt:
        return (
            "A 09:00-11:00 worker 1; B 11:00-14:00 worker 1; "
            "C 11:00-12:00 worker 2; E 12:00-14:00 worker 2; "
            "D 14:00-18:00 worker 1. Final completion 18:00."
        )
    return f"mock benchmark response: {prompt}"


def select_agents(names: Sequence[str] | None = None, *, mode: str = "real") -> list[Any]:
    """Instantiate agent adapters by name."""

    requested = list(names or (("mock",) if mode == "mock" else DEFAULT_AGENT_NAMES))
    agents: list[Any] = []
    for name in requested:
        normalized = name.strip().lower()
        if not normalized:
            continue
        if normalized == "mock" or mode == "mock":
            agent = BenchmarkMockAgent()
            if normalized not in {"mock", "all"} and mode == "mock":
                agent.name = normalized  # type: ignore[misc]
            agents.append(agent)
            continue
        try:
            factory = AGENT_FACTORIES[normalized]
        except KeyError as exc:
            valid = ", ".join([*DEFAULT_AGENT_NAMES, "mock"])
            raise ValueError(f"unknown agent '{name}'. Valid agents: {valid}") from exc
        agents.append(factory())
    if not agents:
        raise ValueError("at least one agent must be selected")
    return agents


def select_tasks(ids: Sequence[str] | None = None) -> list[BenchmarkTask]:
    """Load benchmark tasks by id, preserving caller order."""

    if not ids:
        return all_tasks()
    selected = [get_task(task_id.strip()) for task_id in ids if task_id.strip()]
    if not selected:
        raise ValueError("at least one task must be selected")
    return selected


def run_benchmark(
    *,
    agents: Sequence[Any],
    tasks: Sequence[BenchmarkTask],
    iterations: int = 1,
    run_kwargs: dict[str, Any] | None = None,
) -> MetricsCollector:
    """Run agents x tasks x iterations and return the populated collector."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not agents:
        raise ValueError("agents must not be empty")
    if not tasks:
        raise ValueError("tasks must not be empty")

    collector = MetricsCollector()
    kwargs = dict(run_kwargs or {})
    for iteration in range(1, iterations + 1):
        for agent in agents:
            for task in tasks:
                _run_one(collector, agent, task, iteration, kwargs)
    return collector


def _run_one(
    collector: MetricsCollector,
    agent: Any,
    task: BenchmarkTask,
    iteration: int,
    run_kwargs: dict[str, Any],
) -> None:
    """Run one agent/task pair and append a normalized result."""

    started = time.perf_counter()
    agent_name = str(getattr(agent, "name", agent.__class__.__name__))
    model = getattr(agent, "model", None)
    output = ""
    usage: Any = None
    error: str | None = None
    success = False

    try:
        response = agent.run(task.prompt, **run_kwargs)
        elapsed = time.perf_counter() - started
        if isinstance(response, dict):
            agent_name = str(response.get("agent") or agent_name)
            model_value = response.get("model", model)
            model = str(model_value) if model_value is not None else None
            output = str(response.get("output", ""))
            usage = response.get("usage")
        else:
            output = str(response)
        success = task.validate(output)
        if not success:
            error = "validation failed"
    except Exception as exc:  # pragma: no cover - live SDK/CLI failures vary
        elapsed = time.perf_counter() - started
        error = f"{exc.__class__.__name__}: {exc}"

    collector.record(
        agent=agent_name,
        task_id=task.id,
        task_name=task.name,
        model=model,
        output=output,
        latency_seconds=elapsed,
        usage=usage,
        success=success,
        error=error,
        metadata={"iteration": iteration, "category": task.category},
    )


def _split_csv(values: str | None) -> list[str] | None:
    if values is None:
        return None
    return [value.strip() for value in values.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the agent SDK comparison benchmark across agents, tasks, and iterations.",
    )
    parser.add_argument(
        "--agents",
        help="Comma-separated agents to benchmark: openai, anthropic, goose, gemma, mock. Defaults to all real agents.",
    )
    parser.add_argument(
        "--tasks",
        help=f"Comma-separated task ids to run. Defaults to all tasks: {', '.join(task_ids())}.",
    )
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per agent/task pair. Default: 1.")
    parser.add_argument(
        "--mode",
        choices=("real", "mock"),
        default="real",
        help="Benchmark mode. 'mock' avoids API/CLI calls and validates orchestration only.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON file for benchmark results.")
    parser.add_argument("--summary", action="store_true", help="Print only aggregate summary JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        agents = select_agents(_split_csv(args.agents), mode=args.mode)
        tasks = select_tasks(_split_csv(args.tasks))
        collector = run_benchmark(agents=agents, tasks=tasks, iterations=args.iterations)
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 2

    payload = {"summary": collector.summary(), "results": collector.as_dicts()}
    if args.output:
        export_results(payload["results"], args.output, summary=payload["summary"])

    print(json.dumps(payload["summary"] if args.summary else payload, indent=2))
    return 0 if payload["summary"]["failures"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
