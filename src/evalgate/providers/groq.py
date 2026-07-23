"""LLM judge backed by the free-tier Groq API (OpenAI-compatible)."""

import asyncio
import os
from typing import Any, cast

import httpx

from evalgate.providers.base import JUDGE_MAX_TOKENS, ProviderResponse, as_int

GROQ_TIMEOUT_S = 60.0
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MAX_RETRIES = 5  # free tier ~30 RPM; backoff on 429/503 paces us under it
GROQ_BACKOFF_BASE_S = 2.0
_RETRYABLE_STATUS = frozenset({429, 503})


class GroqError(RuntimeError):
    """Raised when the Groq API call fails or returns an unusable body."""


def parse_groq_response(payload: dict[str, object]) -> ProviderResponse:
    """Turn a Groq (OpenAI-shaped) chat completion into a ProviderResponse.

    Free-tier usage is not billed, so cost_usd is 0.0; tokens come from `usage`.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GroqError(f"groq returned no choices: {payload!r}")
    first = cast(list[object], choices)[0]
    if not isinstance(first, dict):
        raise GroqError(f"groq choice is not an object: {first!r}")
    message = cast(dict[str, object], first).get("message")
    if not isinstance(message, dict):
        raise GroqError(f"groq choice has no message: {first!r}")
    content = cast(dict[str, object], message).get("content")
    if not isinstance(content, str) or not content:
        raise GroqError(f"groq message has no text content: {first!r}")

    usage = payload.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        usage_fields = cast(dict[str, object], usage)
        input_tokens = as_int(usage_fields.get("prompt_tokens"))
        output_tokens = as_int(usage_fields.get("completion_tokens"))
    return ProviderResponse(
        text=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=0.0,
    )


class GroqProvider:
    """LLM judge backed by the free-tier Groq API.

    Reads GROQ_API_KEY. Forces JSON output and temperature 0 for a deterministic
    judge; retries 429/503 with backoff to ride under the free-tier rate limit.
    """

    name = "groq"

    def __init__(
        self,
        model: str = GROQ_DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_s: float = GROQ_TIMEOUT_S,
        base_url: str = GROQ_BASE_URL,
        client: httpx.AsyncClient | None = None,
        max_retries: int = GROQ_MAX_RETRIES,
        backoff_base_s: float = GROQ_BACKOFF_BASE_S,
    ) -> None:
        self.model = model
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise GroqError("GROQ_API_KEY not set")
        self._key = key
        self._base_url = base_url
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    def _request_body(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": JUDGE_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }

    async def complete(self, system: str, user: str) -> ProviderResponse:
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._key}"}
        body = self._request_body(system, user)
        for attempt in range(self._max_retries + 1):
            response = await self._client.post(url, headers=headers, json=body)
            if response.status_code == 200:
                return parse_groq_response(response.json())
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                await asyncio.sleep(self._backoff_base_s * (2**attempt))
                continue
            raise GroqError(f"groq HTTP {response.status_code}: {response.text[:200]}")
        raise GroqError("groq retries exhausted")  # pragma: no cover

    async def aclose(self) -> None:
        await self._client.aclose()
