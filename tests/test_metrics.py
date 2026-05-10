"""Tests for benchmark metrics collection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.collector import BenchmarkResult, MetricsCollector, TokenUsage
from tasks.basic_tasks import BenchmarkTask


class FakeClock:
    def __init__(self) -> None:
        self.values = iter([10.0, 12.5])

    def __call__(self) -> float:
        return next(self.values)


class FakeAgent:
    name = "fake-agent"

    def run(self, prompt: str):
        return {
            "agent": self.name,
            "model": "fake-model",
            "output": f"done: {prompt}",
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        }


def test_token_usage_normalizes_missing_and_total_values() -> None:
    usage = TokenUsage.from_mapping({"prompt_tokens": 3, "completion_tokens": 7})

    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 10
    assert TokenUsage.from_mapping(None).to_dict() == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_metrics_collector_records_result_schema() -> None:
    collector = MetricsCollector()

    result = collector.record(
        agent="openai",
        task_id="code_gen_slugify",
        task_name="Code generation",
        model="gpt-test",
        output="def slugify(): pass",
        latency_seconds=1.25,
        usage={"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
        success=True,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.to_dict() == {
        "agent": "openai",
        "task_id": "code_gen_slugify",
        "task_name": "Code generation",
        "model": "gpt-test",
        "output": "def slugify(): pass",
        "latency_seconds": 1.25,
        "usage": {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
        "success": True,
        "error": None,
        "metadata": {},
    }
    assert collector.results == [result]


def test_metrics_collector_collects_latency_usage_and_success() -> None:
    task = BenchmarkTask(
        id="simple",
        name="Simple task",
        category="unit",
        prompt="hello",
        expected_markers=("done", "hello"),
    )
    collector = MetricsCollector(clock=FakeClock())

    result = collector.collect(FakeAgent(), task)

    assert result.agent == "fake-agent"
    assert result.task_id == "simple"
    assert result.task_name == "Simple task"
    assert result.model == "fake-model"
    assert result.output == "done: hello"
    assert result.latency_seconds == 2.5
    assert result.usage.total_tokens == 10
    assert result.success is True
    assert result.error is None


def test_metrics_collector_marks_failures_and_summarizes() -> None:
    collector = MetricsCollector()
    collector.record(
        agent="a",
        task_id="t1",
        task_name="Task 1",
        output="ok",
        latency_seconds=1.0,
        usage={"total_tokens": 5},
        success=True,
    )
    collector.record(
        agent="a",
        task_id="t2",
        task_name="Task 2",
        output="bad",
        latency_seconds=3.0,
        usage={"total_tokens": 7},
        success=False,
        error="validation failed",
    )

    assert collector.summary() == {
        "total_runs": 2,
        "successes": 1,
        "failures": 1,
        "success_rate": 0.5,
        "total_latency_seconds": 4.0,
        "average_latency_seconds": 2.0,
        "total_tokens": 12,
    }
