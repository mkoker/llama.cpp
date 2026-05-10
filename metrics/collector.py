"""Result schema and metrics collector for agent benchmark runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage values for one agent run."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_mapping(cls, usage: Mapping[str, Any] | Any | None) -> "TokenUsage":
        """Create TokenUsage from SDK-style dicts or objects."""

        if usage is None:
            return cls()

        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("prompt_tokens")
            if prompt_tokens is None:
                prompt_tokens = usage.get("input_tokens")
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is None:
                completion_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        prompt_tokens = _optional_int(prompt_tokens)
        completion_tokens = _optional_int(completion_tokens)
        total_tokens = _optional_int(total_tokens)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def to_dict(self) -> dict[str, int | None]:
        """Return JSON-serializable token usage."""

        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """JSON-serializable result schema for one agent/task run."""

    agent: str
    task_id: str
    task_name: str
    model: str | None
    output: str
    latency_seconds: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    success: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON/CSV export."""

        return {
            "agent": self.agent,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "model": self.model,
            "output": self.output,
            "latency_seconds": self.latency_seconds,
            "usage": self.usage.to_dict(),
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class MetricsCollector:
    """Collect latency, token usage, and success/failure for benchmark runs."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.perf_counter
        self.results: list[BenchmarkResult] = []

    def record(
        self,
        *,
        agent: str,
        task_id: str,
        task_name: str,
        output: str,
        latency_seconds: float,
        success: bool,
        usage: Mapping[str, Any] | Any | None = None,
        model: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Append one benchmark result and return it."""

        result = BenchmarkResult(
            agent=agent,
            task_id=task_id,
            task_name=task_name,
            model=model,
            output=output,
            latency_seconds=float(latency_seconds),
            usage=TokenUsage.from_mapping(usage),
            success=bool(success),
            error=error,
            metadata=dict(metadata or {}),
        )
        self.results.append(result)
        return result

    def collect(self, agent: Any, task: Any, **run_kwargs: Any) -> BenchmarkResult:
        """Run one agent against one task and collect normalized metrics.

        The task object is expected to expose id, name, prompt, and validate().
        Agent failures are captured as failed results instead of escaping, so the
        benchmark harness can continue running remaining agents/tasks.
        """

        started = self._clock()
        agent_name = str(getattr(agent, "name", agent.__class__.__name__))
        model: str | None = getattr(agent, "model", None)
        output = ""
        usage: Mapping[str, Any] | Any | None = None
        error: str | None = None
        success = False

        try:
            response = agent.run(task.prompt, **run_kwargs)
            elapsed = self._clock() - started
            if isinstance(response, Mapping):
                agent_name = str(response.get("agent") or agent_name)
                model_value = response.get("model", model)
                model = str(model_value) if model_value is not None else None
                output = str(response.get("output", ""))
                usage = response.get("usage")
            else:
                output = str(response)
            success = bool(task.validate(output))
            if not success:
                error = "validation failed"
        except Exception as exc:  # pragma: no cover - exercised by harness use
            elapsed = self._clock() - started
            error = f"{exc.__class__.__name__}: {exc}"

        return self.record(
            agent=agent_name,
            task_id=str(task.id),
            task_name=str(task.name),
            model=model,
            output=output,
            latency_seconds=elapsed,
            usage=usage,
            success=success,
            error=error,
        )

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return all collected results as plain dicts."""

        return [result.to_dict() for result in self.results]

    def summary(self) -> dict[str, int | float]:
        """Return aggregate success, latency, and token metrics."""

        total_runs = len(self.results)
        successes = sum(1 for result in self.results if result.success)
        failures = total_runs - successes
        total_latency = sum(result.latency_seconds for result in self.results)
        total_tokens = sum(result.usage.total_tokens or 0 for result in self.results)

        return {
            "total_runs": total_runs,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total_runs if total_runs else 0.0,
            "total_latency_seconds": total_latency,
            "average_latency_seconds": total_latency / total_runs if total_runs else 0.0,
            "total_tokens": total_tokens,
        }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = ["BenchmarkResult", "MetricsCollector", "TokenUsage"]
