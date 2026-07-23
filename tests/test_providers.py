import json

import httpx
import pytest

from evalgate.providers import (
    JUDGE_MODEL,
    PRICING_PER_MTOK,
    ClaudeCliError,
    ClaudeCliProvider,
    GeminiError,
    GeminiProvider,
    GroqError,
    GroqProvider,
    Provider,
    build_claude_argv,
    compute_cost_usd,
    parse_cli_envelope,
    parse_gemini_response,
    parse_groq_response,
)


def _envelope(**overrides: object) -> str:
    base: dict[str, object] = {
        "is_error": False,
        "result": '{"faithfulness": {"score": 5, "rationale": "ok"}}',
        "total_cost_usd": 0.21,
        "usage": {"input_tokens": 29555, "output_tokens": 26},
    }
    base.update(overrides)
    return json.dumps(base)


def test_judge_model_has_pricing() -> None:
    assert JUDGE_MODEL in PRICING_PER_MTOK


def test_cost_computed_from_pricing_table() -> None:
    cost = compute_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
    assert cost == 5.00 + 25.00


def test_cost_scales_with_tokens() -> None:
    assert compute_cost_usd("claude-haiku-4-5", 500_000, 0) == 0.50


def test_unknown_model_costs_zero() -> None:
    assert compute_cost_usd("some-future-model", 1_000_000, 1_000_000) == 0.0


def test_parse_cli_envelope_extracts_result_tokens_and_cost() -> None:
    response = parse_cli_envelope(_envelope())
    assert response.text == '{"faithfulness": {"score": 5, "rationale": "ok"}}'
    assert response.input_tokens == 29555
    assert response.output_tokens == 26
    assert response.cost_usd == 0.21


def test_parse_cli_envelope_missing_usage_defaults_tokens_to_zero() -> None:
    response = parse_cli_envelope(_envelope(usage=None))
    assert response.input_tokens == 0
    assert response.output_tokens == 0


def test_parse_cli_envelope_raises_on_error_flag() -> None:
    with pytest.raises(ClaudeCliError):
        parse_cli_envelope(_envelope(is_error=True, result="rate limited"))


def test_parse_cli_envelope_raises_on_non_json() -> None:
    with pytest.raises(ClaudeCliError):
        parse_cli_envelope("not json at all")


def test_parse_cli_envelope_raises_when_result_not_string() -> None:
    with pytest.raises(ClaudeCliError):
        parse_cli_envelope(_envelope(result={"unexpected": "shape"}))


def test_claude_cli_argv_carries_stripping_flags() -> None:
    argv = build_claude_argv("claude", "sonnet", "SYS")
    assert argv[:2] == ["claude", "-p"]
    assert "--strict-mcp-config" in argv
    assert "--exclude-dynamic-system-prompt-sections" in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--system-prompt") + 1] == "SYS"


def test_claude_cli_provider_satisfies_provider_protocol() -> None:
    assert isinstance(ClaudeCliProvider(executable="claude"), Provider)


def test_claude_cli_raises_when_executable_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def which_missing(_: str) -> str | None:
        return None

    monkeypatch.setattr("evalgate.providers.claude_cli.shutil.which", which_missing)
    with pytest.raises(ClaudeCliError):
        ClaudeCliProvider()


def _gemini_body(text: str) -> dict[str, object]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 812, "candidatesTokenCount": 40},
    }


def test_parse_gemini_response_extracts_text_and_tokens() -> None:
    response = parse_gemini_response(_gemini_body('{"faithfulness": {"score": 4}}'))
    assert response.text == '{"faithfulness": {"score": 4}}'
    assert response.input_tokens == 812
    assert response.output_tokens == 40
    assert response.cost_usd == 0.0  # free tier


def test_parse_gemini_response_raises_on_no_candidates() -> None:
    with pytest.raises(GeminiError):
        parse_gemini_response({"promptFeedback": {"blockReason": "SAFETY"}})


def test_parse_gemini_response_raises_on_empty_parts() -> None:
    with pytest.raises(GeminiError):
        parse_gemini_response({"candidates": [{"content": {"parts": []}}]})


def test_gemini_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiError):
        GeminiProvider()


def _gemini_with(handler: httpx.MockTransport) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        client=httpx.AsyncClient(transport=handler),
        backoff_base_s=0.0,
    )


async def test_gemini_complete_posts_and_parses() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["key"] = request.url.params.get("key")
        return httpx.Response(200, json=_gemini_body('{"correctness": {"score": 5}}'))

    result = await _gemini_with(httpx.MockTransport(handler)).complete("SYS", "USER")
    assert result.text == '{"correctness": {"score": 5}}'
    assert seen["key"] == "test-key"
    assert seen["system_instruction"] == {"parts": [{"text": "SYS"}]}


async def test_gemini_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_gemini_body('{"relevance": {"score": 3}}'))

    result = await _gemini_with(httpx.MockTransport(handler)).complete("SYS", "USER")
    assert calls["n"] == 2
    assert result.text == '{"relevance": {"score": 3}}'


async def test_gemini_raises_on_persistent_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    with pytest.raises(GeminiError):
        await _gemini_with(httpx.MockTransport(handler)).complete("SYS", "USER")


def test_gemini_provider_satisfies_provider_protocol() -> None:
    assert isinstance(GeminiProvider(api_key="test-key"), Provider)


def _groq_body(text: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 72, "completion_tokens": 25},
    }


def test_parse_groq_response_extracts_text_and_tokens() -> None:
    response = parse_groq_response(_groq_body('{"relevance": {"score": 4}}'))
    assert response.text == '{"relevance": {"score": 4}}'
    assert response.input_tokens == 72
    assert response.output_tokens == 25
    assert response.cost_usd == 0.0  # free tier


def test_parse_groq_response_raises_on_no_choices() -> None:
    with pytest.raises(GroqError):
        parse_groq_response({"choices": []})


def test_parse_groq_response_raises_on_missing_content() -> None:
    with pytest.raises(GroqError):
        parse_groq_response({"choices": [{"message": {"role": "assistant"}}]})


def test_groq_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError):
        GroqProvider()


def _groq_with(handler: httpx.MockTransport) -> GroqProvider:
    return GroqProvider(
        api_key="test-key",
        client=httpx.AsyncClient(transport=handler),
        backoff_base_s=0.0,
    )


async def test_groq_complete_posts_bearer_and_parses() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_groq_body('{"correctness": {"score": 5}}'))

    result = await _groq_with(httpx.MockTransport(handler)).complete("SYS", "USER")
    assert result.text == '{"correctness": {"score": 5}}'
    assert seen["auth"] == "Bearer test-key"
    assert seen["response_format"] == {"type": "json_object"}


async def test_groq_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_groq_body('{"faithfulness": {"score": 3}}'))

    result = await _groq_with(httpx.MockTransport(handler)).complete("SYS", "USER")
    assert calls["n"] == 2
    assert result.text == '{"faithfulness": {"score": 3}}'


async def test_groq_raises_on_persistent_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(GroqError):
        await _groq_with(httpx.MockTransport(handler)).complete("SYS", "USER")


def test_groq_provider_satisfies_provider_protocol() -> None:
    assert isinstance(GroqProvider(api_key="test-key"), Provider)
