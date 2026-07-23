"""Provider protocol, shared types, and the default Anthropic judge."""

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from anthropic import AsyncAnthropic

JUDGE_MODEL = "claude-opus-4-8"
LLM_TIMEOUT_S = 30.0
JUDGE_MAX_TOKENS = 1024

# USD per million tokens (input, output); source: platform.claude.com pricing.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@runtime_checkable
class Provider(Protocol):
    name: str
    model: str

    async def complete(self, system: str, user: str) -> ProviderResponse: ...


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_PER_MTOK.get(model)
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def as_int(value: object) -> int:
    """Coerce an unknown JSON value to int, defaulting to 0 — shared by the
    envelope/response parsers of the non-Anthropic providers."""
    return value if isinstance(value, int) else 0


class AnthropicProvider:
    """Anthropic Messages API. Model pinned; no sampling params (rejected on 4.8)."""

    name = "anthropic"

    def __init__(
        self, model: str = JUDGE_MODEL, api_key: str | None = None
    ) -> None:
        self.model = model
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=LLM_TIMEOUT_S,
        )

    async def complete(self, system: str, user: str) -> ProviderResponse:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=JUDGE_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return ProviderResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=compute_cost_usd(
                self.model, response.usage.input_tokens, response.usage.output_tokens
            ),
        )
