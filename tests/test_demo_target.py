"""The seeded demo target must fail exactly its constructed flaw set."""

import asyncio
from pathlib import Path

from evalgate.adapters.callable import CallableAdapter
from evalgate.loader import load_dataset
from evalgate.runner import run_cases
from evalgate.scorers.rules import score_case
from examples.demo_target import (
    OFFTOPIC_ANSWERS,
    UNGROUNDED_ANSWERS,
    WRONG_ANSWERS,
    WRONG_TOOL_PLANS,
    answer,
)

GOLDEN = Path("datasets/golden/golden_v1.jsonl")
RULE_FAIL_IDS = (
    set(WRONG_ANSWERS) | set(OFFTOPIC_ANSWERS) | set(WRONG_TOOL_PLANS)
)


def test_seeded_flaws_fail_and_clean_cases_pass() -> None:
    cases = load_dataset(GOLDEN)
    results = asyncio.run(run_cases(cases, CallableAdapter(answer), retries=0))
    failed_ids = {
        score.case_id
        for case, result in zip(cases, results, strict=True)
        for score in score_case(case, result)
        if score.passed is False
    }
    assert failed_ids == RULE_FAIL_IDS


def test_ungrounded_answers_pass_rule_checks() -> None:
    """The judge-only flaw class: rule checks must NOT catch these."""
    assert not (set(UNGROUNDED_ANSWERS) & RULE_FAIL_IDS)


def test_expected_accuracy_is_stable() -> None:
    cases = load_dataset(GOLDEN)
    results = asyncio.run(run_cases(cases, CallableAdapter(answer), retries=0))
    scores = [
        score
        for case, result in zip(cases, results, strict=True)
        for score in score_case(case, result)
    ]
    accuracy = sum(score.value for score in scores) / len(scores)
    assert round(accuracy, 4) == round(52 / 60, 4)
