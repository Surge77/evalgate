import json

import httpx

from evalgate.adapters.base import Adapter
from evalgate.adapters.http import HttpAdapter
from evalgate.schemas import Case

URL = "http://target.test/answer"


def make_case() -> Case:
    return Case.model_validate(
        {"id": "c1", "category": "qa", "input": {"q": "x"}, "context": ["ctx"]}
    )


def adapter_with(handler: httpx.MockTransport) -> HttpAdapter:
    return HttpAdapter(URL, client=httpx.AsyncClient(transport=handler))


async def test_posts_case_and_parses_output() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"output": "42", "tool_calls": ["search"], "cost_usd": 0.01},
        )

    result = await adapter_with(httpx.MockTransport(handler)).run(make_case())
    assert seen == {"id": "c1", "input": {"q": "x"}, "context": ["ctx"]}
    assert result.output == "42"
    assert result.tool_calls == ["search"]
    assert result.cost_usd == 0.01
    assert result.error is None
    assert result.latency_ms > 0


async def test_minimal_response_needs_only_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": "ok"})

    result = await adapter_with(httpx.MockTransport(handler)).run(make_case())
    assert result.output == "ok"
    assert result.tool_calls == []
    assert result.cost_usd == 0.0


async def test_http_error_status_becomes_error_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    result = await adapter_with(httpx.MockTransport(handler)).run(make_case())
    assert result.error == "HTTP 500"
    assert result.output == ""


async def test_transport_failure_becomes_error_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    result = await adapter_with(httpx.MockTransport(handler)).run(make_case())
    assert result.error is not None and "http error" in result.error


async def test_non_json_body_becomes_error_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    result = await adapter_with(httpx.MockTransport(handler)).run(make_case())
    assert result.error is not None and "invalid JSON" in result.error


async def test_missing_output_field_becomes_error_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "wrong key"})

    result = await adapter_with(httpx.MockTransport(handler)).run(make_case())
    assert result.error is not None and "missing 'output'" in result.error


def test_satisfies_adapter_protocol() -> None:
    adapter = HttpAdapter(URL)
    assert isinstance(adapter, Adapter)
    assert adapter.name == URL
