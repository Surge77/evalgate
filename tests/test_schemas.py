import pytest
from pydantic import ValidationError

from evalgate.schemas import (
    CASE_SCHEMA_VERSION,
    AgentResult,
    Baseline,
    Case,
    Check,
    CheckType,
    RunReport,
    Score,
)


def make_case(**overrides: object) -> Case:
    data: dict[str, object] = {
        "id": "case-001",
        "category": "qa",
        "input": {"question": "What is 2+2?"},
        "reference": "4",
        "checks": [{"type": "exact", "value": "4"}],
    }
    data.update(overrides)
    return Case.model_validate(data)


class TestCase_:
    def test_round_trips_through_json(self) -> None:
        case = make_case()
        restored = Case.model_validate_json(case.model_dump_json())
        assert restored == case

    def test_defaults_schema_version_and_empty_collections(self) -> None:
        case = Case.model_validate(
            {"id": "c", "category": "qa", "input": {"q": "x"}}
        )
        assert case.schema_version == CASE_SCHEMA_VERSION
        assert case.context == []
        assert case.checks == []
        assert case.tags == []
        assert case.reference is None

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            make_case(referance="typo")

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            make_case(id="")

    def test_rejects_unknown_check_type(self) -> None:
        with pytest.raises(ValidationError):
            make_case(checks=[{"type": "fuzzy", "value": "4"}])


class TestCheck:
    def test_all_check_types_parse(self) -> None:
        for check_type in CheckType:
            check = Check.model_validate(
                {"type": check_type.value, "value": "x"}
            )
            assert check.type is check_type

    def test_numeric_tolerance_accepts_tolerance(self) -> None:
        check = Check.model_validate(
            {"type": "numeric_tolerance", "value": "23.9", "tolerance": 0.1}
        )
        assert check.tolerance == 0.1


class TestAgentResult:
    def test_defaults(self) -> None:
        result = AgentResult(case_id="c", output="hello")
        assert result.error is None
        assert result.cost_usd == 0.0
        assert result.latency_ms == 0.0
        assert result.tool_calls == []

    def test_error_result_allows_empty_output(self) -> None:
        result = AgentResult(case_id="c", output="", error="boom")
        assert result.error == "boom"


class TestScore:
    def test_value_bounded_zero_to_one(self) -> None:
        with pytest.raises(ValidationError):
            Score(case_id="c", scorer="rule:exact", value=1.5)
        with pytest.raises(ValidationError):
            Score(case_id="c", scorer="rule:exact", value=-0.1)

    def test_passed_derived_semantics(self) -> None:
        score = Score(case_id="c", scorer="rule:exact", value=1.0, passed=True)
        assert score.passed is True


class TestRunReportAndBaseline:
    def test_run_report_round_trips(self) -> None:
        report = RunReport(
            run_id="run-1",
            dataset="datasets/golden/sample.jsonl",
            target="stub",
            results=[AgentResult(case_id="c", output="4")],
            scores=[Score(case_id="c", scorer="rule:exact", value=1.0, passed=True)],
            metrics={"accuracy": 1.0},
        )
        restored = RunReport.model_validate_json(report.model_dump_json())
        assert restored == report

    def test_baseline_round_trips(self) -> None:
        baseline = Baseline(run_id="run-1", metrics={"accuracy": 0.9})
        restored = Baseline.model_validate_json(baseline.model_dump_json())
        assert restored == baseline
