# evalgate

**pytest for AI quality.** Run a golden dataset through your LLM app, grade every answer
(rule checks + LLM-as-judge), get a scorecard — and block the PR in CI if quality drops.

> Status: **pre-v1 scaffold.** Build plan in [`docs/PLAN.md`](docs/PLAN.md).

## Why

LLM apps break silently: a prompt tweak, a model swap, a retrieval change — and suddenly the
bot hallucinates 8% more. Nobody notices until users do. evalgate makes AI quality a number
and puts a regression gate on it, the same way tests gate normal code.

## How it works

```
golden dataset (JSONL) ─▶ your app (http / python adapter) ─▶ scorers ─▶ scorecard ─▶ CI gate
                                                    rule checks + LLM-as-judge      exit 1 on regression
```

1. **Golden dataset** — versioned cases: input, reference answer, context, checks.
2. **Adapters** — point evalgate at any target: HTTP endpoint or Python callable.
3. **Scorers** — deterministic rule checks (free) + LLM-as-judge for faithfulness/correctness
   (cached, calibrated against human labels).
4. **Scorecard** — accuracy, faithfulness, cost, latency p50/p95; per-case detail; baseline diff.
   Terminal, markdown (PR comment), and static HTML.
5. **Gate** — `evalgate gate` exits non-zero when scores drop vs the promoted baseline. Wire it
   into GitHub Actions; bad changes never merge.

## Numbers (measured at v1 — placeholders until then)

- golden cases: _N_
- judge–human agreement: _X%_ on calibrated subset
- gate demo: blocks a seeded prompt-regression PR
- full run: _$Z_, _T_ min

## Stack

Python 3.11+ · Pydantic v2 · Typer · httpx · Anthropic SDK (judge) · SQLite · rich

## Dev

```sh
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt
pytest          # unit tests — no network, providers stubbed
pyright         # strict
```
