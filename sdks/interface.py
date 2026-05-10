"""Common agent interface for benchmark SDK adapters.

Concrete adapters implement ``run`` and may inherit ``complete``/``__call__``
for the normalized benchmark contract used by the harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, TypedDict, runtime_checkable


class UsageDict(TypedDict):
    """Normalized token usage fields shared by all agents."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class AgentResult(TypedDict, total=False):
    """Normalized result shape returned by AgentInterface.run."""

    agent: str
    model: str | None
    output: str
    latency_seconds: float
    usage: UsageDict
    raw_response_id: str | None
    stderr: str
    command: list[str]
    base_url: str


@runtime_checkable
class AgentLike(Protocol):
    """Structural protocol for existing adapter classes."""

    name: str

    def run(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Execute one benchmark prompt and return normalized data."""
        ...

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Execute one prompt and return only the text output."""
        ...


class AgentInterface(ABC):
    """Abstract base class for benchmark agent adapters.

    Subclasses must implement ``run`` and return a mapping with at least an
    ``output`` field. ``complete`` and ``__call__`` provide common convenience
    behavior for harness code.
    """

    name: str

    @abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Execute one benchmark prompt and return normalized data."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Execute one prompt and return only the text output."""

        return str(self.run(prompt, **kwargs)["output"])

    def __call__(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Alias to run so agents can be invoked as callables."""

        return self.run(prompt, **kwargs)


__all__ = ["AgentInterface", "AgentLike", "AgentResult", "UsageDict"]
