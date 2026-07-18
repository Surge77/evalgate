from evalgate.report import aggregate_metrics
from evalgate.schemas import AgentResult, Score


def result(case_id: str, **overrides: object) -> AgentResult:
    return AgentResult.model_validate(
        {"case_id": case_id, "output": "x", **overrides}
    )


def score(case_id: str, scorer: str, value: float) -> Score:
    return Score(case_id=case_id, scorer=scorer, value=value)


def test_accuracy_is_rule_pass_rate() -> None:
    metrics = aggregate_metrics(
        [result("a"), result("b")],
        [score("a", "rule:exact", 1.0), score("b", "rule:exact", 0.0)],
    )
    assert metrics["accuracy"] == 0.5


def test_judge_dimensions_average_separately() -> None:
    metrics = aggregate_metrics(
        [result("a"), result("b")],
        [
            score("a", "judge:faithfulness", 1.0),
            score("b", "judge:faithfulness", 0.5),
            score("a", "judge:relevance", 0.25),
        ],
    )
    assert metrics["faithfulness"] == 0.75
    assert metrics["relevance"] == 0.25
    assert "accuracy" not in metrics


def test_error_rate_latency_and_cost() -> None:
    metrics = aggregate_metrics(
        [
            result("a", latency_ms=100.0, cost_usd=0.01),
            result("b", latency_ms=300.0, cost_usd=0.02, error="boom", output=""),
        ],
        [],
    )
    assert metrics["error_rate"] == 0.5
    assert metrics["mean_latency_ms"] == 200.0
    assert round(metrics["total_cost_usd"], 4) == 0.03


def test_empty_inputs_yield_empty_metrics() -> None:
    assert aggregate_metrics([], []) == {}
