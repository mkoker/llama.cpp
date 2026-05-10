import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdks.anthropic_agent import AnthropicAgent


class FakeAnthropicClient:
    def __init__(self):
        self.messages = self
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            id="msg_test",
            model=kwargs["model"],
            content=[SimpleNamespace(type="text", text="anthropic reply")],
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


def test_anthropic_agent_normalizes_message_response():
    client = FakeAnthropicClient()
    agent = AnthropicAgent(model="claude-test", client=client)

    result = agent.run("say hi", max_tokens=12)

    assert result["agent"] == "anthropic"
    assert result["model"] == "claude-test"
    assert result["output"] == "anthropic reply"
    assert result["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert result["raw_response_id"] == "msg_test"
    assert client.last_kwargs["messages"] == [{"role": "user", "content": "say hi"}]
    assert client.last_kwargs["system"]
    assert client.last_kwargs["max_tokens"] == 12


def test_complete_returns_text_only():
    agent = AnthropicAgent(client=FakeAnthropicClient())

    assert agent.complete("hello") == "anthropic reply"
