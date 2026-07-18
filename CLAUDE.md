# evalgate — Project Instructions

> Overrides `~/.claude/CLAUDE.md` and all `~/.claude/rules/*` where they conflict.

## What this is
Offline eval harness for LLM apps/agents: golden dataset → rule scorers + LLM-as-judge →
per-run scorecard (accuracy, faithfulness, cost, latency) → **CI regression gate** that blocks
PRs when quality drops vs baseline. CLI-first, no web server. Full plan: `docs/PLAN.md`.

## Stack
- **Language:** Python 3.11+. Type gate: `pyright` (strict; config `pyrightconfig.json`). Tests: `pytest` (config `pytest.ini`).
- **Core deps:** Pydantic v2 (schemas), Typer (CLI), httpx (async), Anthropic SDK (judge), rich (terminal output), SQLite via stdlib (run history).
- **Layout:** `src/evalgate/` package, `tests/` mirrors it, `datasets/golden/` JSONL cases, `artifacts/` generated per run (gitignored).
- **Deps:** `requirements.txt` (runtime) + `requirements-dev.txt`, exact pins. Secrets via `.env` (never commit).

## Hard rules (project)
- Never commit or push directly to `main` after the initial scaffold — branch + PR.
- **No real LLM/network calls in unit tests.** Judge and adapters are stubbed at the provider
  interface. Live-call tests go in `tests/live/` and are excluded from CI.
- Every external HTTP request needs a timeout (default 30s for LLM calls — they are slow; max 120s).
- Judge responses and target responses are **cached by content hash** — re-scoring must never
  re-call an LLM for an unchanged (case, config) pair.
- Scorers are pure functions: `(Case, AgentResult) -> Score`. No I/O inside scorers; the runner owns I/O.
- Dataset schema changes require a version bump in the case schema and a loader migration.
- File size limit 300 lines (global rule applies).

## Agent routing
Same pinned table as firstApply: `planner` (features >3 files), `python-reviewer` (all code review),
`tdd-guide`, `build-error-resolver` (pyright/build), `debugger` (runtime bugs), `security-reviewer`
(HTTP/env keys/subprocess), `dependency-auditor` (any new dep). Never use non-Python language
reviewers or `fastapi-reviewer` (no FastAPI here).
