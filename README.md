# evalgate

**pytest for AI quality.** An offline eval harness for LLM apps and agents: run a golden
dataset through your target, grade every answer with rule checks and an LLM judge, get a
scorecard, and **fail CI when quality drops vs a promoted baseline**.

```
golden dataset (JSONL) ─▶ your app (python / http adapter) ─▶ rule scorers + LLM judge
                        ─▶ scorecard (rich / md / html) ─▶ SQLite history
                        ─▶ gate vs baseline ─▶ CI exit 0/1
```

LLM apps break silently: a prompt tweak, a model swap, a retrieval change — and suddenly
the bot hallucinates 8% more. evalgate makes AI quality a number and puts a regression
gate on it, the same way tests gate normal code.

## Quickstart

```sh
pip install -e .

# 1. Run the golden dataset through a target (python callable or http URL)
evalgate run --target examples.demo_target:answer --dataset datasets/golden/golden_v1.jsonl

# 2. Promote a good run as the baseline (commits evalgate.baseline.json)
evalgate baseline promote

# 3. Gate every PR: exit 1 if quality drops vs baseline
evalgate gate
```

Targets are anything that answers a case: a `module:function` spec (sync or async,
returning `str` or an `AgentResult`) or an `http(s)://` URL that accepts
`POST {id, input, context}` and returns `{"output": ...}`.

## Measured numbers (2026-07-19, this repo's demo pipeline)

| Number | Value | How measured |
|---|---|---|
| Golden cases | **60** across 5 categories (qa, extraction, math, classification, tool_plan) | `datasets/golden/golden_v1.jsonl` |
| Demo target accuracy | **0.8667** (52/60 rule checks pass; 12 flaws seeded by construction, 8 rule-detectable) | `evalgate run` on `examples.demo_target:answer` |
| Full offline run wall time | **1.3 s** (rule checks only, no cache) | timed CLI run |
| Offline run cost | **$0.00** | no LLM calls on the rule-check path |
| Gate blocks seeded regression | **yes** — sabotaged target drops accuracy 0.8667 → 0.0, `evalgate gate` exits 1 | verified locally; covered by `tests/test_cli.py` |
| Judge–human agreement | **pending live run** — see Calibration below | requires `ANTHROPIC_API_KEY` |
| Judge cost per full run | **pending live run** (estimate ≈ $0.35 for 60 cases on `claude-opus-4-8`) | reported as `judge_cost_usd` in run metrics |

## The judge

Rule checks catch wrong answers; they cannot catch *ungrounded* ones. Four of the demo
target's seeded flaws pass every rule check while inventing facts absent from the
context — only the judge's **faithfulness** dimension catches them.

The judge (pinned `claude-opus-4-8`, 30 s timeout) scores **faithfulness**,
**correctness**, and **relevance** 1–5, normalized to 0–1. Responses are cached by
content hash of (provider, model, prompt version, case, answer) — re-scoring an
unchanged pair never re-calls the LLM.

```sh
# needs ANTHROPIC_API_KEY in the environment or .env
evalgate run --target examples.demo_target:answer \
  --dataset datasets/golden/golden_v1.jsonl --judge-provider anthropic
```

### Calibration

A judge without a measured agreement number is vibes. 30 of the 60 cases are labeled in
`datasets/calibration/labels_v1.jsonl` (labels derived from the demo target's
construction ground truth — the flaw set is known by design). After a judged run:

```sh
evalgate calibrate datasets/calibration/labels_v1.jsonl
# → per-dimension agreement + overall %, tolerance ±1 on the 1-5 scale
```

## The gate

Thresholds live in `evalgate.toml`:

```toml
[gate]
baseline = "evalgate.baseline.json"

[gate.max_drop]          # fail if a metric drops more than this vs baseline
accuracy = 0.05

[gate.min]               # fail if below this floor (works without a baseline)
accuracy = 0.75

[gate.warn_increase_pct] # warn (never fail) on cost creep
total_cost_usd = 25.0
```

`evalgate gate` exits 1 on regression (2 on config error) and emits
`::error`/`::warning` annotations under GitHub Actions. A metric named in the config but
missing from the run **fails closed**. This repo's own CI runs the demo eval and gates
every PR — see `.github/workflows/ci.yml`.

## Commands

| Command | Does |
|---|---|
| `evalgate run --target T [--dataset D] [--judge-provider P] [--limit N] [--no-cache]` | run + score, writes report JSON + SQLite history under `artifacts/` |
| `evalgate report [--run-id ID] [--format rich\|md\|html] [--output F]` | render a stored run, with vs-baseline deltas |
| `evalgate baseline promote [RUN_ID]` | write the run's metrics to `evalgate.baseline.json` |
| `evalgate gate [--config evalgate.toml]` | threshold check, CI exit code |
| `evalgate calibrate LABELS.jsonl` | judge–human agreement % |

## Dataset schema

One JSON object per line (`schema_version` 1):

```json
{"id": "qa-001", "category": "qa",
 "input": {"question": "..."}, "context": ["..."], "reference": "...",
 "checks": [{"type": "contains", "value": "14"}], "tags": ["policy"]}
```

Check types: `exact`, `contains`, `regex`, `json_schema` (subset: type/required/
properties), `numeric_tolerance` (any number in the output within `tolerance`),
`tool_sequence` (exact ordered match against `AgentResult.tool_calls`).

## Stack

Python 3.11+ · Pydantic v2 · Typer · httpx · Anthropic SDK (judge) · SQLite · rich

## Development

```sh
python -m venv .venv
pip install -r requirements.txt -r requirements-dev.txt -e .
pytest          # 163 tests, no network; scorers and gate at 100% coverage
pyright         # strict
```

Live-call tests belong in `tests/live/` (excluded from CI). Full design doc:
[`docs/PLAN.md`](docs/PLAN.md).
