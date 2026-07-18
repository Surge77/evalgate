"""LLM-as-judge scorer: faithfulness, correctness, relevance.

Pure helpers (prompt build, parse, normalize) are separated from the
provider I/O so tests can cover them without any LLM call.
"""

import asyncio
import json
from typing import cast

from evalgate.cache import ResponseCache, cache_key
from evalgate.providers import Provider
from evalgate.schemas import AgentResult, Case, Score

JUDGE_PROMPT_VERSION = "v1"
DIMENSIONS = ("faithfulness", "correctness", "relevance")
MIN_JUDGE_SCORE = 1
MAX_JUDGE_SCORE = 5

JUDGE_SYSTEM = """You are a strict evaluator of AI assistant answers.
Score the answer on three dimensions, each an integer 1-5:
- faithfulness: is the answer grounded in the provided context? (5 = fully grounded, 1 = contradicts or invents facts; if no context is given, judge against the reference)
- correctness: does the answer match the reference answer? (5 = equivalent, 1 = wrong)
- relevance: does the answer address the question asked? (5 = directly, 1 = off-topic)

Respond with ONLY a JSON object, no prose, no code fences:
{"faithfulness": {"score": <1-5>, "rationale": "<one sentence>"},
 "correctness": {"score": <1-5>, "rationale": "<one sentence>"},
 "relevance": {"score": <1-5>, "rationale": "<one sentence>"}}"""


class JudgeError(Exception):
    """Raised when the judge response cannot be parsed into scores."""


def build_user_prompt(case: Case, result: AgentResult) -> str:
    context = "\n".join(f"- {chunk}" for chunk in case.context) or "(none)"
    return (
        f"QUESTION:\n{json.dumps(case.input, ensure_ascii=False)}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"REFERENCE ANSWER:\n{case.reference or '(none)'}\n\n"
        f"ANSWER TO EVALUATE:\n{result.output}"
    )


def parse_judgment(text: str) -> dict[str, tuple[int, str]]:
    """Parse the judge JSON into {dimension: (score, rationale)}."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        data: object = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge returned invalid JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise JudgeError(f"judge JSON is not an object: {text[:200]!r}")
    judgment = cast(dict[str, object], data)

    parsed: dict[str, tuple[int, str]] = {}
    for dimension in DIMENSIONS:
        entry = judgment.get(dimension)
        if not isinstance(entry, dict) or "score" not in entry:
            raise JudgeError(f"judge JSON missing dimension {dimension!r}")
        fields = cast(dict[str, object], entry)
        score = fields["score"]
        if not isinstance(score, int) or not (
            MIN_JUDGE_SCORE <= score <= MAX_JUDGE_SCORE
        ):
            raise JudgeError(f"{dimension} score out of range: {score!r}")
        parsed[dimension] = (score, str(fields.get("rationale", "")))
    return parsed


def normalize_score(score: int) -> float:
    """Map the 1-5 judge scale onto the harness-wide 0-1 scale."""
    return (score - MIN_JUDGE_SCORE) / (MAX_JUDGE_SCORE - MIN_JUDGE_SCORE)


def _scores_from_judgment(
    case_id: str, judgment: dict[str, tuple[int, str]]
) -> list[Score]:
    return [
        Score(
            case_id=case_id,
            scorer=f"judge:{dimension}",
            value=normalize_score(score),
            detail=rationale,
        )
        for dimension, (score, rationale) in judgment.items()
    ]


async def judge_case(
    case: Case,
    result: AgentResult,
    provider: Provider,
    cache: ResponseCache | None = None,
) -> tuple[list[Score], float]:
    """Judge one answer. Returns (scores, judge_cost_usd). Cache hits cost 0."""
    if result.error is not None:
        return [
            Score(
                case_id=case.id,
                scorer=f"judge:{dimension}",
                value=0.0,
                passed=False,
                detail=f"target error: {result.error}",
            )
            for dimension in DIMENSIONS
        ], 0.0

    key = cache_key(
        "judge",
        provider.name,
        provider.model,
        JUDGE_PROMPT_VERSION,
        case.model_dump_json(),
        result.output,
    )
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return _scores_from_judgment(case.id, parse_judgment(hit)), 0.0

    response = await provider.complete(JUDGE_SYSTEM, build_user_prompt(case, result))
    judgment = parse_judgment(response.text)
    if cache is not None:
        cache.put(key, response.text)
    return _scores_from_judgment(case.id, judgment), response.cost_usd


async def judge_all(
    cases: list[Case],
    results: list[AgentResult],
    provider: Provider,
    cache: ResponseCache | None = None,
    concurrency: int = 4,
) -> tuple[list[Score], float]:
    """Judge every (case, result) pair. Returns (all scores, total judge cost)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case: Case, result: AgentResult) -> tuple[list[Score], float]:
        async with semaphore:
            return await judge_case(case, result, provider, cache)

    outcomes = await asyncio.gather(
        *(bounded(case, result) for case, result in zip(cases, results, strict=True))
    )
    scores = [score for case_scores, _ in outcomes for score in case_scores]
    total_cost = sum(cost for _, cost in outcomes)
    return scores, total_cost


def agreement_rate(
    judge_scores: dict[tuple[str, str], int],
    human_scores: dict[tuple[str, str], int],
    tolerance: int = 1,
) -> float:
    """Share of human-labeled (case, dimension) pairs where the judge agrees
    within `tolerance` points on the 1-5 scale. The calibration number."""
    common = judge_scores.keys() & human_scores.keys()
    if not common:
        raise ValueError("no overlapping (case_id, dimension) labels to compare")
    agreed = sum(
        1 for k in common if abs(judge_scores[k] - human_scores[k]) <= tolerance
    )
    return agreed / len(common)
