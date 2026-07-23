"""LLM judge providers. The stub point for all tests.

Each backend implements the `Provider` protocol; the judge scorer and cache
depend only on that protocol, so backends are peers selectable at the CLI.
"""

from evalgate.providers.base import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    LLM_TIMEOUT_S,
    PRICING_PER_MTOK,
    AnthropicProvider,
    Provider,
    ProviderResponse,
    as_int,
    compute_cost_usd,
)
from evalgate.providers.claude_cli import (
    ClaudeCliError,
    ClaudeCliProvider,
    build_claude_argv,
    parse_cli_envelope,
)
from evalgate.providers.gemini import (
    GeminiError,
    GeminiProvider,
    parse_gemini_response,
)
from evalgate.providers.groq import GroqError, GroqProvider, parse_groq_response

__all__ = [
    "JUDGE_MAX_TOKENS",
    "JUDGE_MODEL",
    "LLM_TIMEOUT_S",
    "PRICING_PER_MTOK",
    "AnthropicProvider",
    "Provider",
    "ProviderResponse",
    "as_int",
    "compute_cost_usd",
    "ClaudeCliError",
    "ClaudeCliProvider",
    "build_claude_argv",
    "parse_cli_envelope",
    "GeminiError",
    "GeminiProvider",
    "parse_gemini_response",
    "GroqError",
    "GroqProvider",
    "parse_groq_response",
]
