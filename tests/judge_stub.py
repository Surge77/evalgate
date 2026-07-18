"""Canned judge provider — the offline stand-in for the Anthropic judge."""

import json

from evalgate.providers import ProviderResponse

PERFECT_JUDGMENT = json.dumps(
    {
        "faithfulness": {"score": 5, "rationale": "grounded"},
        "correctness": {"score": 5, "rationale": "matches reference"},
        "relevance": {"score": 5, "rationale": "on topic"},
    }
)


class StaticJudgeProvider:
    name = "stub"
    model = "stub-judge"

    def __init__(self, text: str = PERFECT_JUDGMENT, cost_usd: float = 0.001) -> None:
        self._text = text
        self._cost_usd = cost_usd
        self.calls = 0

    async def complete(self, system: str, user: str) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            text=self._text, input_tokens=100, output_tokens=50, cost_usd=self._cost_usd
        )


provider = StaticJudgeProvider()
