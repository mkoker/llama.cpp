"""Repeatable benchmark tasks for the agent SDK comparison harness.

Each task returns a deterministic prompt plus lightweight validation metadata so
future harness code can run the same work across every SDK adapter. The tasks are
kept offline-safe: the agent being benchmarked may use web/tools if it has them,
but constructing the task never performs network or filesystem side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


Validator = Callable[[str], bool]


@dataclass(frozen=True)
class BenchmarkTask:
    """One repeatable agent benchmark task."""

    id: str
    name: str
    category: str
    prompt: str
    expected_markers: tuple[str, ...] = field(default_factory=tuple)
    validator: Validator | None = None

    def validate(self, output: str) -> bool:
        """Return True when output satisfies the task's basic success check."""

        text = output.strip()
        if not text:
            return False
        if self.validator is not None:
            return self.validator(text)
        return all(marker.lower() in text.lower() for marker in self.expected_markers)


def web_search_task() -> BenchmarkTask:
    """Ask for a current-facts web lookup with source URLs."""

    return BenchmarkTask(
        id="web_search_current_ai_news",
        name="Web search: current AI news with citations",
        category="web_search",
        prompt=(
            "Find two recent AI model or agent SDK announcements from the last 30 days. "
            "Return exactly two bullets. Each bullet must include the announcement, "
            "the organization, date, and a source URL."
        ),
        expected_markers=("http",),
        validator=lambda output: output.count("http") >= 2 and output.count("\n") >= 1,
    )


def code_generation_task() -> BenchmarkTask:
    """Ask for small deterministic Python code generation."""

    return BenchmarkTask(
        id="code_gen_slugify",
        name="Code generation: Python slugify helper",
        category="code_generation",
        prompt=(
            "Write a single Python function named slugify(text: str) -> str. "
            "It must lowercase input, replace runs of non-alphanumeric characters "
            "with one hyphen, strip leading/trailing hyphens, and include three assert "
            "examples. Return only code."
        ),
        expected_markers=("def slugify", "assert"),
    )


def file_read_write_task() -> BenchmarkTask:
    """Ask for file IO steps without relying on actual local side effects."""

    return BenchmarkTask(
        id="file_read_write_summary",
        name="File read/write: transform JSON lines into summary file",
        category="file_read_write",
        prompt=(
            "Assume a workspace contains input/events.jsonl with one JSON object per line: "
            "{\"team\": str, \"score\": int}. Describe the exact Python code needed to read "
            "that file, compute total score by team, and write output/team_scores.json. "
            "Return only a complete Python script."
        ),
        expected_markers=("events.jsonl", "team_scores.json", "json"),
    )


def data_parsing_task() -> BenchmarkTask:
    """Ask for deterministic extraction from semi-structured text."""

    return BenchmarkTask(
        id="data_parse_invoice_text",
        name="Data parsing: extract invoice fields",
        category="data_parsing",
        prompt=(
            "Parse this invoice text and return strict JSON with keys invoice_id, vendor, "
            "invoice_date, due_date, line_items, subtotal, tax, total. line_items must be a "
            "list of objects with description, quantity, unit_price, amount. Text:\n"
            "Invoice INV-1042 | Vendor: Northwind Tools | Date: 2026-05-01 | Due: 2026-05-31\n"
            "Items: 3 x Router Bit @ 12.50 = 37.50; 2 x Safety Goggles @ 18.00 = 36.00\n"
            "Subtotal 73.50 | Tax 4.41 | Total 77.91"
        ),
        expected_markers=("INV-1042", "Northwind", "77.91"),
    )


def multi_step_reasoning_task() -> BenchmarkTask:
    """Ask for a constrained planning/reasoning answer with an auditable result."""

    return BenchmarkTask(
        id="reasoning_project_schedule",
        name="Multi-step reasoning: dependency-aware schedule",
        category="multi_step_reasoning",
        prompt=(
            "A project has tasks A=2h, B=3h after A, C=1h after A, D=4h after B and C, "
            "E=2h after C. Two workers are available, tasks are non-preemptive, and work starts "
            "at 09:00. Build the shortest valid schedule. Return start/end time for each task, "
            "assigned worker, and final completion time."
        ),
        expected_markers=("09:00", "A", "D"),
    )


def all_tasks() -> list[BenchmarkTask]:
    """Return the canonical ordered task set for benchmark runs."""

    return [
        web_search_task(),
        code_generation_task(),
        file_read_write_task(),
        data_parsing_task(),
        multi_step_reasoning_task(),
    ]


def get_task(task_id: str) -> BenchmarkTask:
    """Look up a benchmark task by id."""

    for task in all_tasks():
        if task.id == task_id:
            return task
    raise KeyError(f"unknown benchmark task: {task_id}")


def task_ids(tasks: Iterable[BenchmarkTask] | None = None) -> list[str]:
    """Return stable task ids for display/config validation."""

    return [task.id for task in (tasks if tasks is not None else all_tasks())]


__all__ = [
    "BenchmarkTask",
    "Validator",
    "all_tasks",
    "code_generation_task",
    "data_parsing_task",
    "file_read_write_task",
    "get_task",
    "multi_step_reasoning_task",
    "task_ids",
    "web_search_task",
]
