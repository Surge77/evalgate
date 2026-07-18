# evalgate — Build Plan

_Decided 2026-07-18. Source: Notion "Projects to build" → 2026 priority #1 (Agent Eval Harness)._

## Vision

pytest for AI quality. A CLI that runs a golden dataset through any LLM app, grades every
answer (rule checks + LLM-as-judge), produces a scorecard, and **fails CI when quality drops
vs baseline**. Turns "the AI seems fine" into "the AI scores 87% and no change ships if that drops."

**Non-goals (v1):** web dashboard, live/online eval, model benchmarking (MMLU-style), prompt
management, multi-tenant anything. CLI + generated artifacts only.

## Architecture

```
datasets/golden/*.jsonl ──▶ loader ──▶ Case[]
                                        │
                              runner (async, bounded concurrency, cache)
                                        │  Adapter.run(case) ──▶ AgentResult
                                        ▼      (http | python-callable)
                              scorers: rule (pure fns) + judge (LLM, cached)
                                        │
                                        ▼
                     RunReport ──▶ SQLite history ──▶ scorecard (rich / md / html)
                                        │
                                        ▼
                          gate: compare vs baseline ──▶ exit 0/1 (CI)
```

Module map (`src/evalgate/`):

| Module | Responsibility |
|--------|---------------|
| `schemas.py` | Pydantic: `Case`, `Check`, `AgentResult`, `Score`, `RunReport`, `Baseline` |
| `loader.py` | JSONL dataset load + validation; schema-version check |
| `adapters/` | `base.py` protocol; `http.py`; `callable.py`; per-target configs |
| `scorers/rules.py` | exact/regex/contains/json_schema/numeric-tolerance/tool-sequence — pure fns |
| `scorers/judge.py` | LLM-as-judge: faithfulness, correctness, relevance; structured output |
| `providers.py` | thin LLM provider interface (Anthropic first) — the stub point for tests |
| `cache.py` | content-hash response cache (SQLite) for target + judge calls |
| `runner.py` | async orchestration, retries/backoff, resume |
| `report.py` | aggregate metrics, per-case detail, baseline diff; rich/md/html render |
| `store.py` | SQLite run history; baseline promotion |
| `gate.py` | threshold rules → pass/fail; GitHub-Actions-friendly output |
| `cli.py` | Typer: `run`, `score`, `report`, `baseline promote`, `gate` |

## Milestones

| # | Days | Deliverable | Done when |
|---|------|-------------|-----------|
| M1 | 1–2 | schemas + loader + adapter protocol + callable adapter | loader round-trips `sample.jsonl`; pyright strict green |
| M2 | 3–4 | rule scorers + runner + cache | `evalgate run` scores a stub target end-to-end offline |
| M3 | 5–7 | judge scorers + calibration | judge scores cached; human-labeled subset → agreement % reported |
| M4 | 8–9 | scorecard + baseline diff + store | `evalgate report` renders rich + md + html; diff vs promoted baseline |
| M5 | 10–11 | CI gate + first real target | GitHub Action fails a PR that degrades faithfulness; `ai-pr-review-bot` wired via http adapter |
| M6 | 12–14 | README numbers + polish | README states real measured numbers; demo run committed as artifact |

Each milestone = branch + PR. TDD throughout (tests before implementation, 80%+ coverage;
scorers and gate at 100% — they are the business logic).

## Judge design (M3)

- Judge prompt returns structured JSON: `{score: 1-5, rationale: str}` per dimension.
- Dimensions: **faithfulness** (grounded in provided context?), **correctness** (matches
  reference?), **relevance** (answers the question?).
- **Calibration:** ~30 cases hand-labeled → report judge–human agreement %. This number goes
  in the README; without it the judge is vibes.
- Judge model pinned + temperature 0; judge calls cached by (case, answer, prompt-version) hash.

## Gate rules (M5)

Configurable thresholds in `evalgate.toml` (per-repo):
- fail if mean faithfulness drops > X points vs baseline
- fail if accuracy (rule-check pass rate) drops > Y%
- warn (not fail) if cost per run rises > Z%
Baseline = explicitly promoted run (`evalgate baseline promote <run-id>`), stored in repo as JSON.

## README numbers (targets to measure, not invent)

- N golden cases (target ≥ 60 across ≥ 3 categories)
- judge–human agreement ≥ 85% on calibration subset
- gate blocks a seeded regression PR (demo: prompt sabotage → CI red)
- full run cost $Z and wall time T min (measure at M6)

## Risks

- **Judge reliability** → calibration subset is mandatory, not optional; disagreement > 15% means judge prompt iteration before M4.
- **Flaky targets** → retries + cache; runs resumable by run-id.
- **Cost creep during dev** → cache-first design; `--limit N` flag for dev runs; stub provider in tests.
