"""Gemma/local OpenAI-compatible wrapper for the benchmark harness.

The local Gemma service is exposed through an OpenAI-compatible chat
completions endpoint, typically llama.cpp/vLLM/Ollama-compatible infrastructure.
Importing this module is offline-safe: no client is constructed and no network
request is made until ``run`` is called.
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


def _default_base_url() -> str:
    """Return the configured local OpenAI-compatible endpoint."""

    return (
        os.getenv("GEMMA_BASE_URL")
        or os.getenv("LOCAL_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://192.168.1.169:8080/v1"
    )


@dataclass
class GemmaAgent:
    """Small adapter for local Gemma through OpenAI-compatible chat completions."""

    model: str = "gemma-4-31b-it"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = 1024
    timeout: float | None = 120.0
    system_prompt: str = "You are a precise benchmark agent. Return only the requested result."
    client: Any | None = field(default=None, repr=False)

    name: str = "gemma"

    def _client(self) -> Any:
        """Return a lazily-created OpenAI-compatible client."""

        if self.client is not None:
            return self.client
        if OpenAI is None:
            raise RuntimeError("openai package is not installed") from _OPENAI_IMPORT_ERROR

        self.client = OpenAI(
            api_key=self.api_key or os.getenv("GEMMA_API_KEY") or os.getenv("LOCAL_LLM_API_KEY") or "local",
            base_url=self.base_url or _default_base_url(),
            timeout=self.timeout,
        )
        return self.client

    def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Execute one local Gemma prompt and return normalized benchmark data."""

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

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        output = "" if message is None else (getattr(message, "content", None) or "")
        usage = getattr(response, "usage", None)

        return {
            "agent": self.name,
            "model": getattr(response, "model", self.model),
            "output": output,
            "latency_seconds": elapsed,
            "usage": self._usage_dict(usage),
            "raw_response_id": getattr(response, "id", None),
            "base_url": str(self.base_url or _default_base_url()),
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


__all__ = ["GemmaAgent"]
