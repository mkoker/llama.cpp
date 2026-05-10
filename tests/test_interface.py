"""Tests for the common agent interface."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdks.interface import AgentInterface


def test_agent_interface_requires_run_implementation() -> None:
    with pytest.raises(TypeError):
        AgentInterface()


def test_agent_interface_accepts_minimal_concrete_agent() -> None:
    class MinimalAgent(AgentInterface):
        name = "minimal"

        def run(self, prompt: str, **kwargs):
            return {
                "agent": self.name,
                "model": "mock",
                "output": prompt.upper(),
                "latency_seconds": 0.0,
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "raw_response_id": None,
            }

    agent = MinimalAgent()

    assert agent.complete("hello") == "HELLO"
    assert agent("hello")["output"] == "HELLO"
