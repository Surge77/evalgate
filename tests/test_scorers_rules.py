import pytest

from evalgate.schemas import AgentResult, Case
from evalgate.scorers.rules import score_case


def make_case(checks: list[dict[str, object]]) -> Case:
    return Case.model_validate(
        {"id": "c1", "category": "qa", "input": {"q": "x"}, "checks": checks}
    )


def make_result(output: str, **overrides: object) -> AgentResult:
    return AgentResult.model_validate(
        {"case_id": "c1", "output": output, **overrides}
    )


def single(case: Case, result: AgentResult) -> tuple[bool | None, str]:
    scores = score_case(case, result)
    assert len(scores) == 1
    return scores[0].passed, scores[0].detail


@pytest.mark.parametrize(
    ("check", "output", "expect_pass"),
    [
        ({"type": "exact", "value": "4"}, "4", True),
        ({"type": "exact", "value": "4"}, " 4 ", True),
        ({"type": "exact", "value": "4"}, "5", False),
        ({"type": "contains", "value": "14"}, "within 14 days", True),
        ({"type": "contains", "value": "14"}, "within 15 days", False),
        ({"type": "regex", "value": r"\b14 days\b"}, "in 14 days.", True),
        ({"type": "regex", "value": r"^\d+$"}, "days: 14", False),
        ({"type": "numeric_tolerance", "value": "23.9"}, "Total is 23.9", True),
        (
            {"type": "numeric_tolerance", "value": "23.9", "tolerance": 0.2},
            "Total: $24.05",
            True,
        ),
        ({"type": "numeric_tolerance", "value": "23.9"}, "Total: 25", False),
        ({"type": "numeric_tolerance", "value": "23.9"}, "no numbers here", False),
        ({"type": "tool_sequence", "value": "search, fetch"}, "", False),
        ({"type": "json_schema", "value": "{\"type\": \"object\"}"}, "{}", True),
        ({"type": "json_schema", "value": "{\"type\": \"object\"}"}, "not json", False),
    ],
)
def test_check_outcomes(
    check: dict[str, object], output: str, expect_pass: bool
) -> None:
    passed, detail = single(make_case([check]), make_result(output))
    assert passed is expect_pass
    if not expect_pass:
        assert detail


def test_tool_sequence_matches_result_tool_calls() -> None:
    case = make_case([{"type": "tool_sequence", "value": "search, fetch"}])
    result = make_result("done", tool_calls=["search", "fetch"])
    passed, _ = single(case, result)
    assert passed is True


def test_tool_sequence_order_matters() -> None:
    case = make_case([{"type": "tool_sequence", "value": "search, fetch"}])
    result = make_result("done", tool_calls=["fetch", "search"])
    passed, _ = single(case, result)
    assert passed is False


def test_json_schema_required_and_nested_properties() -> None:
    schema = (
        '{"type": "object", "required": ["merchant", "total"],'
        ' "properties": {"total": {"type": "number"}}}'
    )
    case = make_case([{"type": "json_schema", "value": schema}])
    ok = make_result('{"merchant": "BigMart", "total": 23.9}')
    assert single(case, ok)[0] is True
    missing = make_result('{"merchant": "BigMart"}')
    passed, detail = single(case, missing)
    assert passed is False and "total" in detail
    wrong_type = make_result('{"merchant": "BigMart", "total": "23.9"}')
    passed, detail = single(case, wrong_type)
    assert passed is False and "number" in detail


def test_json_schema_boolean_is_not_number() -> None:
    schema = '{"type": "object", "properties": {"total": {"type": "number"}}}'
    case = make_case([{"type": "json_schema", "value": schema}])
    passed, detail = single(case, make_result('{"total": true}'))
    assert passed is False and "boolean" in detail


def test_json_schema_invalid_schema_fails_with_detail() -> None:
    case = make_case([{"type": "json_schema", "value": "{broken"}])
    passed, detail = single(case, make_result("{}"))
    assert passed is False and "schema" in detail


def test_json_schema_unsupported_type_fails() -> None:
    case = make_case([{"type": "json_schema", "value": '{"type": "null"}'}])
    passed, detail = single(case, make_result("null"))
    assert passed is False and "unsupported" in detail


def test_target_error_fails_every_check() -> None:
    case = make_case(
        [{"type": "exact", "value": "4"}, {"type": "contains", "value": "4"}]
    )
    result = make_result("", error="timeout")
    scores = score_case(case, result)
    assert [score.passed for score in scores] == [False, False]
    assert all("target error" in score.detail for score in scores)


def test_scorer_names_follow_rule_prefix() -> None:
    case = make_case([{"type": "contains", "value": "4"}])
    scores = score_case(case, make_result("4"))
    assert scores[0].scorer == "rule:contains"


def test_no_checks_yields_no_scores() -> None:
    case = make_case([])
    assert score_case(case, make_result("anything")) == []


def test_purity_inputs_not_mutated() -> None:
    case = make_case([{"type": "exact", "value": "4"}])
    result = make_result("4")
    before_case, before_result = case.model_copy(deep=True), result.model_copy(deep=True)
    score_case(case, result)
    assert case == before_case and result == before_result
