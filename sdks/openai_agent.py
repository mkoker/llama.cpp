"""OpenAI SDK wrapper for the agent SDK comparison harness.

The wrapper is intentionally light: importing it does not require an API key and
constructing the SDK client is deferred until the first real request. This keeps
cron gates and unit tests offline-safe while still exposing a usable runtime
adapter for benchmark tasks.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency gate covers this
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR: ImportError | None = exc
else:
    _OPENAI_IMPORT_ERROR = None


@dataclass
class OpenAIAgent:
    """Small OpenAI chat-completions adapter.

    Parameters are deliberately generic so the benchmark runner can instantiate
    this class from config.yaml later without needing another compatibility
    layer.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float | None = 60.0
    system_prompt: str = "You are a precise benchmark agent. Return only the requested result."
    client: Any | None = field(default=None, repr=False)

    name: str = "openai"

    def _client(self) -> Any:
        """Return a lazily-created OpenAI client.

        Raises a clear error only when a live request is attempted without the
        dependency installed. The official SDK will handle missing credentials
        with its own standard AuthenticationError/ValueError path.
        """

        if self.client is not None:
            return self.client
        if OpenAI is None:
            raise RuntimeError("openai package is not installed") from _OPENAI_IMPORT_ERROR

        kwargs: dict[str, Any] = {"timeout": self.timeout}
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if key:
            kwargs["api_key"] = key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)
        return self.client

    def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a single prompt and return normalized benchmark data."""

        started = time.perf_counter()
        response = self._client().chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=[
                {"role": "system", "content": kwargs.pop("system_prompt", self.system_prompt)},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )
        elapsed = time.perf_counter() - started

        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        return {
            "agent": self.name,
            "model": response.model,
            "output": message.content or "",
            "latency_seconds": elapsed,
            "usage": self._usage_dict(usage),
            "raw_response_id": getattr(response, "id", None),
        }

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Convenience API for callers that only need text."""

        return str(self.run(prompt, **kwargs)["output"])

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return self.run(prompt, **kwargs)

    @staticmethod
    def _usage_dict(usage: Any) -> dict[str, int | None]:
        if usage is None:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        if isinstance(usage, Mapping):
            return {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }


__all__ = ["OpenAIAgent"]
