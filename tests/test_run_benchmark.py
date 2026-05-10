"""Tests for the main benchmark runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmark import BenchmarkMockAgent, run_benchmark, select_agents, select_tasks
from tasks.basic_tasks import BenchmarkTask


def _task(task_id: str, marker: str) -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        name=f"Task {task_id}",
        category="unit",
        prompt=f"prompt {task_id}",
        expected_markers=(marker,),
    )


class RecordingAgent:
    def __init__(self, name: str, marker: str) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.marker = marker
        self.prompts: list[str] = []

    def run(self, prompt: str):
        self.prompts.append(prompt)
        return {
            "agent": self.name,
            "model": self.model,
            "output": f"{self.marker}: {prompt}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }


def test_run_benchmark_orchestrates_agents_tasks_and_iterations() -> None:
    agents = [RecordingAgent("alpha", "ok"), RecordingAgent("beta", "ok")]
    tasks = [_task("one", "ok"), _task("two", "ok")]

    collector = run_benchmark(agents=agents, tasks=tasks, iterations=2)

    results = collector.as_dicts()
    assert len(results) == 8
    assert [result["metadata"]["iteration"] for result in results] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert {(result["agent"], result["task_id"]) for result in results} == {
        ("alpha", "one"),
        ("alpha", "two"),
        ("beta", "one"),
        ("beta", "two"),
    }
    assert collector.summary()["successes"] == 8


def test_run_benchmark_records_failures_without_stopping() -> None:
    agents = [RecordingAgent("good", "ok"), RecordingAgent("bad", "nope")]
    tasks = [_task("one", "ok")]

    collector = run_benchmark(agents=agents, tasks=tasks, iterations=1)

    assert [result.success for result in collector.results] == [True, False]
    assert collector.results[1].error == "validation failed"


def test_selectors_load_mock_agent_and_named_tasks() -> None:
    agents = select_agents(["mock"], mode="mock")
    tasks = select_tasks(["code_gen_slugify"])

    assert len(agents) == 1
    assert isinstance(agents[0], BenchmarkMockAgent)
    assert [task.id for task in tasks] == ["code_gen_slugify"]


def test_cli_help_mentions_benchmark() -> None:
    completed = subprocess.run(
        [sys.executable, "run_benchmark.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "benchmark" in completed.stdout.lower()
