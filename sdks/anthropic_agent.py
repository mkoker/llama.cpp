"""Anthropic SDK wrapper for the agent SDK comparison harness.

Importing this module is offline-safe: credentials and the official SDK client
are only needed when a live request is executed. The wrapper normalizes Anthropic
Messages API responses into the same shape used by the other benchmark agents.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from anthropic import Anthropic
except ImportError as exc:  # pragma: no cover - dependency gate covers this
    Anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_IMPORT_ERROR: ImportError | None = exc
else:
    _ANTHROPIC_IMPORT_ERROR = None


@dataclass
class AnthropicAgent:
    """Small Anthropic Messages API adapter for benchmark tasks."""

    model: str = "claude-3-5-haiku-latest"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float | None = 60.0
    system_prompt: str = "You are a precise benchmark agent. Return only the requested result."
    client: Any | None = field(default=None, repr=False)

    name: str = "anthropic"

    def _client(self) -> Any:
        """Return a lazily-created Anthropic client."""

        if self.client is not None:
            return self.client
        if Anthropic is None:
            raise RuntimeError("anthropic package is not installed") from _ANTHROPIC_IMPORT_ERROR

        kwargs: dict[str, Any] = {"timeout": self.timeout}
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            kwargs["api_key"] = key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = Anthropic(**kwargs)
        return self.client

    def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a single prompt and return normalized benchmark data."""

        started = time.perf_counter()
        response = self._client().messages.create(
            model=kwargs.pop("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            system=kwargs.pop("system_prompt", self.system_prompt),
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )
        elapsed = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        return {
            "agent": self.name,
            "model": getattr(response, "model", self.model),
            "output": self._content_text(getattr(response, "content", None)),
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
    def _content_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, Mapping):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
                continue
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
        return "".join(parts)

    @staticmethod
    def _usage_dict(usage: Any) -> dict[str, int | None]:
        if usage is None:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        else:
            prompt_tokens = getattr(usage, "input_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "output_tokens", None)
            if completion_tokens is None:
                completion_tokens = getattr(usage, "completion_tokens", None)

        total_tokens: int | None
        if prompt_tokens is None or completion_tokens is None:
            total_tokens = None
        else:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


__all__ = ["AnthropicAgent"]
