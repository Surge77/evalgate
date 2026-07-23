"""Adapter that POSTs cases to an HTTP target."""

import time
from typing import Any, cast

import httpx

from evalgate.schemas import AgentResult, Case

HTTP_TIMEOUT_S = 30.0


class HttpAdapter:
    """POSTs {id, input, context} to a URL; expects JSON with an 'output' field.

    Optional response fields: tool_calls (list[str]), cost_usd (float).
    Transport errors, non-2xx statuses, and malformed bodies become error
    results — one dead endpoint must not abort the run.
    """

    def __init__(
        self,
        url: str,
        timeout_s: float = HTTP_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = url
        self._url = url
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def run(self, case: Case) -> AgentResult:
        payload = {"id": case.id, "input": case.input, "context": case.context}
        start = time.perf_counter()
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            body: Any = response.json()
        except httpx.HTTPStatusError as exc:
            return self._error(case, f"HTTP {exc.response.status_code}", start)
        except httpx.HTTPError as exc:
            return self._error(case, f"http error: {exc}", start)
        except ValueError as exc:
            return self._error(case, f"invalid JSON response: {exc}", start)

        if not isinstance(body, dict) or "output" not in body:
            return self._error(case, "response JSON missing 'output' field", start)
        data = cast(dict[str, Any], body)
        latency_ms = (time.perf_counter() - start) * 1000
        return AgentResult(
            case_id=case.id,
            output=str(data["output"]),
            tool_calls=[str(tool) for tool in data.get("tool_calls", [])],
            cost_usd=float(data.get("cost_usd", 0.0)),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _error(self, case: Case, message: str, start: float) -> AgentResult:
        latency_ms = (time.perf_counter() - start) * 1000
        return AgentResult(
            case_id=case.id, output="", error=message, latency_ms=latency_ms
        )
