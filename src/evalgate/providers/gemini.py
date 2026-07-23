"""LLM judge backed by the free-tier Gemini API — no card required."""

import asyncio
import os
from typing import Any, cast

import httpx

from evalgate.providers.base import JUDGE_MAX_TOKENS, ProviderResponse, as_int

GEMINI_TIMEOUT_S = 60.0
# `-latest` alias routes to the current flash model; pinned/2.0 ids 404 or 429
# for keys created after their free-tier window closed.
GEMINI_DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MAX_RETRIES = 5  # free tier ~15 RPM; backoff on 429/503 paces us under it
GEMINI_BACKOFF_BASE_S = 2.0
_RETRYABLE_STATUS = frozenset({429, 503})


class GeminiError(RuntimeError):
    """Raised when the Gemini API call fails or returns an unusable body."""


def parse_gemini_response(payload: dict[str, object]) -> ProviderResponse:
    """Turn a Gemini generateContent body into a ProviderResponse.

    Free-tier usage is not billed, so cost_usd is 0.0; tokens come from
    usageMetadata for transparency in the report.
    """
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        block = payload.get("promptFeedback")
        raise GeminiError(f"gemini returned no candidates (feedback: {block!r})")
    first = cast(list[object], candidates)[0]
    if not isinstance(first, dict):
        raise GeminiError(f"gemini candidate is not an object: {first!r}")
    content = cast(dict[str, object], first).get("content")
    if not isinstance(content, dict):
        raise GeminiError(f"gemini candidate has no content: {first!r}")
    parts = cast(dict[str, object], content).get("parts")
    if not isinstance(parts, list) or not parts:
        raise GeminiError(f"gemini content has no parts: {first!r}")

    texts: list[str] = []
    for part in cast(list[object], parts):
        if not isinstance(part, dict):
            continue
        value = cast(dict[str, object], part).get("text")
        if isinstance(value, str):
            texts.append(value)
    text = "".join(texts)
    if not text:
        raise GeminiError("gemini content parts contained no text")

    usage = payload.get("usageMetadata")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        usage_fields = cast(dict[str, object], usage)
        input_tokens = as_int(usage_fields.get("promptTokenCount"))
        output_tokens = as_int(usage_fields.get("candidatesTokenCount"))
    return ProviderResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=0.0,
    )


class GeminiProvider:
    """LLM judge backed by the free-tier Gemini API.

    Reads GEMINI_API_KEY (or GOOGLE_API_KEY). Forces JSON output and temperature 0
    for a deterministic judge; retries 429/503 with backoff to ride under the
    free-tier rate limit.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = GEMINI_DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_s: float = GEMINI_TIMEOUT_S,
        base_url: str = GEMINI_BASE_URL,
        client: httpx.AsyncClient | None = None,
        max_retries: int = GEMINI_MAX_RETRIES,
        backoff_base_s: float = GEMINI_BACKOFF_BASE_S,
    ) -> None:
        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not key:
            raise GeminiError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        self._key = key
        self._base_url = base_url
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    def _request_body(self, system: str, user: str) -> dict[str, Any]:
        return {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": JUDGE_MAX_TOKENS,
                "responseMimeType": "application/json",
            },
        }

    async def complete(self, system: str, user: str) -> ProviderResponse:
        url = f"{self._base_url}/models/{self.model}:generateContent"
        body = self._request_body(system, user)
        for attempt in range(self._max_retries + 1):
            response = await self._client.post(
                url, params={"key": self._key}, json=body
            )
            if response.status_code == 200:
                return parse_gemini_response(response.json())
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                await asyncio.sleep(self._backoff_base_s * (2**attempt))
                continue
            raise GeminiError(
                f"gemini HTTP {response.status_code}: {response.text[:200]}"
            )
        raise GeminiError("gemini retries exhausted")  # pragma: no cover

    async def aclose(self) -> None:
        await self._client.aclose()
