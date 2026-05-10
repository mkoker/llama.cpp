"""Deterministic offline mock agent for benchmark harness dry-runs."""

from __future__ import annotations

import json
from typing import Any


class MockAgent:
    """Offline-safe deterministic agent for harness validation."""

    name = "mock"
    model = "benchmark-mock"

    def run(self, prompt: str, **_: Any) -> dict[str, Any]:
        """Return a benchmark-style response mapping for *prompt*."""

        output = mock_output_for_prompt(prompt)
        prompt_count = len(prompt.split())
        completion_count = len(output.split())
        return {
            "agent": self.name,
            "model": self.model,
            "output": output,
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": completion_count,
                "total_tokens": prompt_count + completion_count,
            },
        }


def mock_output_for_prompt(prompt: str) -> str:
    """Return deterministic output that satisfies the built-in task validators."""

    if "two recent AI model" in prompt:
        return (
            "- Example AI announcement, ExampleOrg, 2026-05-01, https://example.com/ai-1\n"
            "- Example agent SDK announcement, SDKOrg, 2026-05-02, https://example.com/ai-2"
        )
    if "slugify" in prompt:
        return """import re

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

assert slugify('Hello, World!') == 'hello-world'
assert slugify(' A  B ') == 'a-b'
assert slugify('x_y') == 'x-y'
""".strip()
    if "events.jsonl" in prompt:
        return """import json
from collections import defaultdict

totals = defaultdict(int)
with open('input/events.jsonl') as f:
    for line in f:
        row = json.loads(line)
        totals[row['team']] += row['score']
with open('output/team_scores.json', 'w') as f:
    json.dump(dict(totals), f, indent=2)
""".strip()
    if "INV-1042" in prompt:
        return json.dumps(
            {
                "invoice_id": "INV-1042",
                "vendor": "Northwind Tools",
                "invoice_date": "2026-05-01",
                "due_date": "2026-05-31",
                "line_items": [
                    {
                        "description": "Router Bit",
                        "quantity": 3,
                        "unit_price": 12.50,
                        "amount": 37.50,
                    },
                    {
                        "description": "Safety Goggles",
                        "quantity": 2,
                        "unit_price": 18.00,
                        "amount": 36.00,
                    },
                ],
                "subtotal": 73.50,
                "tax": 4.41,
                "total": 77.91,
            }
        )
    if "A=2h" in prompt:
        return (
            "A 09:00-11:00 worker 1; B 11:00-14:00 worker 1; "
            "C 11:00-12:00 worker 2; E 12:00-14:00 worker 2; "
            "D 14:00-18:00 worker 1. Final completion 18:00."
        )
    return f"mock benchmark response: {prompt}"


__all__ = ["MockAgent", "mock_output_for_prompt"]
