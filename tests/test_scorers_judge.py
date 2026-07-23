import json
from pathlib import Path

import pytest

from evalgate.cache import ResponseCache
from evalgate.schemas import AgentResult, Case
from evalgate.scorers.judge import (
    DIMENSIONS,
    JudgeError,
    agreement_rate,
    build_user_prompt,
    denormalize_score,
    judge_all,
    judge_case,
    normalize_score,
    parse_judgment,
)
from tests.judge_stub import PERFECT_JUDGMENT, StaticJudgeProvider


def make_case(case_id: str = "c1") -> Case:
    return Case.model_validate(
        {
            "id": case_id,
            "category": "qa",
            "input": {"question": "refund window?"},
            "context": ["Refunds within 14 days."],
            "reference": "14 days.",
        }
    )


def make_result(case_id: str = "c1", output: str = "14 days.") -> AgentResult:
    return AgentResult(case_id=case_id, output=output)


class TestParseJudgment:
    def test_parses_valid_json(self) -> None:
        judgment = parse_judgment(PERFECT_JUDGMENT)
        assert judgment["faithfulness"] == (5, "grounded")
        assert set(judgment) == set(DIMENSIONS)

    def test_strips_code_fences(self) -> None:
        fenced = f"```json\n{PERFECT_JUDGMENT}\n```"
        assert parse_judgment(fenced)["correctness"][0] == 5

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(JudgeError, match="invalid JSON"):
            parse_judgment("not json at all")

    def test_non_object_raises(self) -> None:
        with pytest.raises(JudgeError, match="not an object"):
            parse_judgment("[1, 2, 3]")

    def test_missing_dimension_raises(self) -> None:
        with pytest.raises(JudgeError, match="relevance"):
            parse_judgment('{"faithfulness": {"score": 5}, "correctness": {"score": 5}}')

    def test_out_of_range_score_raises(self) -> None:
        bad = json.loads(PERFECT_JUDGMENT)
        bad["relevance"]["score"] = 6
        with pytest.raises(JudgeError, match="out of range"):
            parse_judgment(json.dumps(bad))

    def test_non_integer_score_raises(self) -> None:
        bad = json.loads(PERFECT_JUDGMENT)
        bad["relevance"]["score"] = "five"
        with pytest.raises(JudgeError, match="out of range"):
            parse_judgment(json.dumps(bad))


def test_normalize_maps_1_to_5_onto_0_to_1() -> None:
    assert normalize_score(1) == 0.0
    assert normalize_score(3) == 0.5
    assert normalize_score(5) == 1.0


def test_denormalize_round_trips_every_judge_score() -> None:
    for raw in range(1, 6):
        assert denormalize_score(normalize_score(raw)) == raw


def test_user_prompt_contains_all_sections() -> None:
    prompt = build_user_prompt(make_case(), make_result())
    assert "refund window?" in prompt
    assert "Refunds within 14 days." in prompt
    assert "14 days." in prompt


def test_user_prompt_empty_context_says_none() -> None:
    case = make_case().model_copy(update={"context": []})
    assert "(none)" in build_user_prompt(case, make_result())


async def test_judge_case_returns_normalized_scores_and_cost() -> None:
    provider = StaticJudgeProvider()
    scores, cost = await judge_case(make_case(), make_result(), provider)
    assert [score.scorer for score in scores] == [
        "judge:faithfulness", "judge:correctness", "judge:relevance",
    ]
    assert all(score.value == 1.0 for score in scores)
    assert cost == 0.001
    assert provider.calls == 1


async def test_judge_case_cache_hit_skips_provider(tmp_path: Path) -> None:
    provider = StaticJudgeProvider()
    with ResponseCache(tmp_path / "cache.db") as cache:
        _, first_cost = await judge_case(make_case(), make_result(), provider, cache)
        scores, second_cost = await judge_case(
            make_case(), make_result(), provider, cache
        )
    assert provider.calls == 1
    assert first_cost == 0.001
    assert second_cost == 0.0
    assert len(scores) == 3


async def test_changed_answer_invalidates_judge_cache(tmp_path: Path) -> None:
    provider = StaticJudgeProvider()
    with ResponseCache(tmp_path / "cache.db") as cache:
        await judge_case(make_case(), make_result(output="14 days."), provider, cache)
        await judge_case(make_case(), make_result(output="30 days."), provider, cache)
    assert provider.calls == 2


async def test_unparseable_judgment_raises_and_is_not_cached(tmp_path: Path) -> None:
    provider = StaticJudgeProvider(text="garbage")
    with ResponseCache(tmp_path / "cache.db") as cache:
        with pytest.raises(JudgeError):
            await judge_case(make_case(), make_result(), provider, cache)
        with pytest.raises(JudgeError):
            await judge_case(make_case(), make_result(), provider, cache)
    assert provider.calls == 2, "failed judgments must not be cached"


async def test_target_error_scores_zero_without_calling_judge() -> None:
    provider = StaticJudgeProvider()
    result = AgentResult(case_id="c1", output="", error="timeout")
    scores, cost = await judge_case(make_case(), result, provider)
    assert all(score.value == 0.0 and score.passed is False for score in scores)
    assert cost == 0.0
    assert provider.calls == 0


async def test_judge_all_aggregates_scores_and_cost() -> None:
    provider = StaticJudgeProvider()
    cases = [make_case("c1"), make_case("c2")]
    results = [make_result("c1"), make_result("c2")]
    scores, total_cost = await judge_all(cases, results, provider)
    assert len(scores) == 6
    assert round(total_cost, 6) == 0.002
    assert provider.calls == 2


class TestAgreementRate:
    def test_within_tolerance_counts_as_agreement(self) -> None:
        judge = {("c1", "faithfulness"): 4, ("c2", "faithfulness"): 1}
        human = {("c1", "faithfulness"): 5, ("c2", "faithfulness"): 4}
        assert agreement_rate(judge, human, tolerance=1) == 0.5

    def test_exact_tolerance_zero(self) -> None:
        judge = {("c1", "correctness"): 4}
        human = {("c1", "correctness"): 4}
        assert agreement_rate(judge, human, tolerance=0) == 1.0

    def test_only_common_keys_compared(self) -> None:
        judge = {("c1", "relevance"): 5, ("c9", "relevance"): 1}
        human = {("c1", "relevance"): 5}
        assert agreement_rate(judge, human) == 1.0

    def test_no_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="no overlapping"):
            agreement_rate({("a", "x"): 1}, {("b", "y"): 1})
