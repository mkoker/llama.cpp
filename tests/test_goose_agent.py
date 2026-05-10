import sys
from pathlib import Path
from subprocess import CompletedProcess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdks.goose_agent import GooseAgent


def test_goose_agent_builds_headless_command():
    agent = GooseAgent(binary="python3", with_builtins=("developer",), extra_args=("--debug",))

    command = agent._command("summarize this")

    assert command == [
        "python3",
        "run",
        "--no-session",
        "--with-builtin",
        "developer",
        "--debug",
        "-t",
        "summarize this",
    ]


def test_goose_agent_normalizes_subprocess_response():
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="goose reply\n", stderr="")

    agent = GooseAgent(binary="python3", model="test-model", runner=fake_runner)

    result = agent.run("say hi", timeout=1)

    assert result["agent"] == "goose"
    assert result["model"] == "test-model"
    assert result["output"] == "goose reply"
    assert result["usage"] == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    assert result["raw_response_id"] is None
    assert result["command"][-2:] == ["-t", "<prompt>"]
    assert calls[0][1]["env"]["GOOSE_MODEL"] == "test-model"
    assert calls[0][1]["timeout"] == 1


def test_complete_returns_text_only():
    def fake_runner(command, **kwargs):
        return CompletedProcess(command, 0, stdout="text only", stderr="")

    agent = GooseAgent(binary="python3", runner=fake_runner)

    assert agent.complete("hello") == "text only"
