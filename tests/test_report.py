from evalgate.report import (
    aggregate_metrics,
    baseline_diff,
    render_html,
    render_markdown,
)
from evalgate.schemas import AgentResult, Baseline, RunReport, Score


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


def make_run_report() -> RunReport:
    return RunReport(
        run_id="r1",
        dataset="d.jsonl",
        target="stub",
        results=[result("a")],
        scores=[
            Score(case_id="a", scorer="rule:exact", value=1.0, passed=True),
            Score(
                case_id="a",
                scorer="rule:contains",
                value=0.0,
                passed=False,
                detail="output does not contain '14'",
            ),
        ],
        metrics={"accuracy": 0.5, "faithfulness": 0.9},
    )


def test_baseline_diff_covers_common_metrics_only() -> None:
    baseline = Baseline(run_id="r0", metrics={"accuracy": 0.75, "cost": 1.0})
    diff = baseline_diff({"accuracy": 0.5, "faithfulness": 0.9}, baseline)
    assert diff == {"accuracy": -0.25}


def test_markdown_contains_metrics_and_failures() -> None:
    md = render_markdown(make_run_report())
    assert "# evalgate run `r1`" in md
    assert "| accuracy | 0.5000 |" in md
    assert "## Failures" in md
    assert "rule:contains" in md


def test_markdown_shows_baseline_delta() -> None:
    baseline = Baseline(run_id="r0", metrics={"accuracy": 0.75})
    md = render_markdown(make_run_report(), baseline)
    assert "-0.2500" in md
    assert "baseline: `r0`" in md


def test_html_is_escaped_and_structured() -> None:
    report = make_run_report()
    report.scores[1].detail = "expected <b>bold</b> & more"
    html = render_html(report)
    assert "<table>" in html
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; more" in html
    assert "accuracy" in html


def test_html_without_failures_says_none() -> None:
    report = make_run_report()
    report.scores = [report.scores[0]]
    assert "<li>none</li>" in render_html(report)
