"""LLM judge backed by the local `claude` CLI — uses Claude Code auth, no API key."""

import asyncio
import json
import shutil
import tempfile
from typing import cast

from evalgate.providers.base import ProviderResponse, as_int

CLAUDE_CLI_TIMEOUT_S = 120.0  # cold subprocess + model latency; well above LLM_TIMEOUT_S
CLAUDE_CLI_DEFAULT_MODEL = "sonnet"
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'


class ClaudeCliError(RuntimeError):
    """Raised when the `claude` CLI subprocess fails or returns unusable output."""


def build_claude_argv(executable: str, model: str, system: str) -> list[str]:
    """Headless `claude` argv with all context-stripping flags: no MCP, no
    dynamic (per-machine) system sections — so judgments carry no local context."""
    return [
        executable,
        "-p",
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system,
        "--strict-mcp-config",
        "--mcp-config", _EMPTY_MCP_CONFIG,
        "--exclude-dynamic-system-prompt-sections",
    ]


def parse_cli_envelope(raw: str) -> ProviderResponse:
    """Turn a `claude -p --output-format json` envelope into a ProviderResponse.

    `cost_usd` is the CLI's reported `total_cost_usd`: a *notional* API-equivalent
    figure inflated by the Claude Code harness prompt carried on every cold call —
    the marginal cost on a subscription is zero.
    """
    try:
        envelope: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClaudeCliError(f"claude CLI returned non-JSON: {raw[:200]!r}") from exc
    if not isinstance(envelope, dict):
        raise ClaudeCliError(f"claude CLI envelope not an object: {raw[:200]!r}")
    env = cast(dict[str, object], envelope)
    if env.get("is_error"):
        raise ClaudeCliError(f"claude CLI reported error: {env.get('result')!r}")
    result = env.get("result")
    if not isinstance(result, str):
        raise ClaudeCliError(f"claude CLI envelope missing string 'result': {raw[:200]!r}")

    input_tokens = 0
    output_tokens = 0
    usage = env.get("usage")
    if isinstance(usage, dict):
        usage_fields = cast(dict[str, object], usage)
        input_tokens = as_int(usage_fields.get("input_tokens"))
        output_tokens = as_int(usage_fields.get("output_tokens"))
    cost = env.get("total_cost_usd")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else 0.0
    return ProviderResponse(
        text=result,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


class ClaudeCliProvider:
    """LLM judge backed by the local `claude` CLI in headless (`-p`) mode.

    Uses the machine's existing Claude Code auth (e.g. a Pro/Max subscription)
    instead of an ANTHROPIC_API_KEY. Each call runs a cold subprocess in a clean
    temp cwd with MCP disabled, so no project context leaks into the judgment.
    """

    name = "claude-cli"

    def __init__(
        self,
        model: str = CLAUDE_CLI_DEFAULT_MODEL,
        timeout_s: float = CLAUDE_CLI_TIMEOUT_S,
        executable: str | None = None,
    ) -> None:
        self.model = model
        self._timeout_s = timeout_s
        resolved = executable or shutil.which("claude")
        if resolved is None:
            raise ClaudeCliError("`claude` CLI not found on PATH")
        self._executable = resolved
        self._cwd: str | None = None

    def _clean_cwd(self) -> str:
        # One reused empty dir (no CLAUDE.md/MCP) for every call. A fresh temp dir
        # per call races the OS on Windows cleanup — the claude child keeps a
        # handle on its cwd (WinError 32 on rmtree). Reuse dodges that entirely.
        if self._cwd is None:
            self._cwd = tempfile.mkdtemp(prefix="evalgate-judge-")
        return self._cwd

    async def complete(self, system: str, user: str) -> ProviderResponse:
        argv = build_claude_argv(self._executable, self.model, system)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._clean_cwd(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(user.encode("utf-8")), timeout=self._timeout_s
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ClaudeCliError(
                f"claude CLI timed out after {self._timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", "replace")[:200]
            raise ClaudeCliError(f"claude CLI exited {proc.returncode}: {detail}")
        return parse_cli_envelope(stdout.decode("utf-8", "replace"))
