import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdks.gemma_agent import GemmaAgent


class FakeGemmaClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            id="cmpl_local",
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="gemma reply"))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )


def test_gemma_agent_normalizes_openai_compatible_response():
    client = FakeGemmaClient()
    agent = GemmaAgent(model="gemma-test", base_url="http://local/v1", client=client)

    result = agent.run("say hi", max_tokens=8)

    assert result["agent"] == "gemma"
    assert result["model"] == "gemma-test"
    assert result["output"] == "gemma reply"
    assert result["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    assert result["raw_response_id"] == "cmpl_local"
    assert result["base_url"] == "http://local/v1"
    assert client.last_kwargs["messages"] == [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": "say hi"},
    ]
    assert client.last_kwargs["max_tokens"] == 8


def test_complete_returns_text_only():
    agent = GemmaAgent(client=FakeGemmaClient())

    assert agent.complete("hello") == "gemma reply"
