"""Goose headless CLI wrapper for the agent SDK comparison harness.

Goose is distributed primarily as a CLI agent rather than a Python SDK. This
adapter keeps imports offline-safe and shells out only when ``run`` is called,
using Goose's non-interactive headless mode: ``goose run -t <prompt>``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class GooseAgent:
    """Small adapter around ``goose run`` for benchmark tasks.

    The wrapper intentionally has no import-time dependency on Goose. A missing
    CLI raises a clear RuntimeError only when a live request is attempted. Tests
    and cron gates can import the module without credentials or local Goose
    configuration.
    """

    model: str | None = None
    binary: str = "goose"
    timeout: float | None = 300.0
    cwd: str | Path | None = None
    env: Mapping[str, str] | None = None
    no_session: bool = True
    with_builtins: Sequence[str] = field(default_factory=tuple)
    extra_args: Sequence[str] = field(default_factory=tuple)
    runner: Runner = field(default=subprocess.run, repr=False)

    name: str = "goose"

    def _command(self, prompt: str, **kwargs: Any) -> list[str]:
        """Build a Goose headless command for one prompt."""

        command = [str(kwargs.pop("binary", self.binary)), "run"]

        no_session = kwargs.pop("no_session", self.no_session)
        if no_session:
            command.append("--no-session")

        builtins = kwargs.pop("with_builtins", self.with_builtins)
        for builtin in builtins:
            command.extend(["--with-builtin", str(builtin)])

        model = kwargs.pop("model", self.model)
        if model:
            # Goose reads provider/model from its config and environment. There
            # is no stable universal model flag across releases, so expose the
            # requested model to child processes for configured wrappers/scripts.
            pass

        extra_args = kwargs.pop("extra_args", self.extra_args)
        command.extend(str(arg) for arg in extra_args)
        command.extend(["-t", prompt])
        return command

    def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Execute Goose headlessly and return normalized benchmark data."""

        command = self._command(prompt, **kwargs)
        binary = command[0]
        if shutil.which(binary) is None and Path(binary).name == binary:
            raise RuntimeError(
                f"goose CLI binary '{binary}' is not installed or not on PATH; "
                "install Goose before running live Goose benchmarks"
            )

        child_env = os.environ.copy()
        if self.env:
            child_env.update({str(k): str(v) for k, v in self.env.items()})
        if self.model:
            child_env.setdefault("GOOSE_MODEL", self.model)
        if kwargs.get("model"):
            child_env["GOOSE_MODEL"] = str(kwargs["model"])
        if kwargs.get("env"):
            child_env.update({str(k): str(v) for k, v in dict(kwargs["env"]).items()})

        started = time.perf_counter()
        completed = self.runner(
            command,
            cwd=str(kwargs.get("cwd", self.cwd)) if kwargs.get("cwd", self.cwd) else None,
            env=child_env,
            text=True,
            capture_output=True,
            timeout=kwargs.get("timeout", self.timeout),
            check=False,
        )
        elapsed = time.perf_counter() - started

        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "goose headless run failed "
                f"with exit code {completed.returncode}: {error or output}"
            )

        return {
            "agent": self.name,
            "model": kwargs.get("model", self.model),
            "output": output,
            "latency_seconds": elapsed,
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "raw_response_id": None,
            "stderr": error,
            "command": self._redacted_command(command),
        }

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Convenience API for callers that only need text."""

        return str(self.run(prompt, **kwargs)["output"])

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return self.run(prompt, **kwargs)

    @staticmethod
    def _redacted_command(command: Sequence[str]) -> list[str]:
        """Return command metadata without leaking the full benchmark prompt."""

        redacted: list[str] = []
        skip_next = False
        for part in command:
            if skip_next:
                redacted.append("<prompt>")
                skip_next = False
                continue
            redacted.append(part)
            if part in {"-t", "--text", "--prompt"}:
                skip_next = True
        return redacted


__all__ = ["GooseAgent"]
