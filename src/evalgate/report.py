"""Aggregate metrics, baseline diff, and markdown/html rendering."""

from html import escape
from statistics import mean

from evalgate.schemas import AgentResult, Baseline, RunReport, Score

RULE_PREFIX = "rule:"
JUDGE_PREFIX = "judge:"


def aggregate_metrics(
    results: list[AgentResult], scores: list[Score]
) -> dict[str, float]:
    """Roll scores and run stats up into named metrics.

    accuracy        — mean over all rule-check scores (pass rate)
    <dimension>     — mean over judge scores per dimension (0-1 scale)
    error_rate      — share of cases whose target errored
    mean_latency_ms — mean target latency
    total_cost_usd  — summed target + judge spend
    """
    metrics: dict[str, float] = {}

    rule_values = [score.value for score in scores if score.scorer.startswith(RULE_PREFIX)]
    if rule_values:
        metrics["accuracy"] = mean(rule_values)

    judge_dims: dict[str, list[float]] = {}
    for score in scores:
        if score.scorer.startswith(JUDGE_PREFIX):
            dimension = score.scorer.removeprefix(JUDGE_PREFIX)
            judge_dims.setdefault(dimension, []).append(score.value)
    for dimension, values in judge_dims.items():
        metrics[dimension] = mean(values)

    if results:
        metrics["error_rate"] = sum(
            1 for result in results if result.error is not None
        ) / len(results)
        metrics["mean_latency_ms"] = mean(result.latency_ms for result in results)
        metrics["total_cost_usd"] = sum(result.cost_usd for result in results)
    return metrics


def baseline_diff(
    metrics: dict[str, float], baseline: Baseline
) -> dict[str, float]:
    """Delta (current - baseline) for every metric present in both."""
    return {
        name: metrics[name] - baseline.metrics[name]
        for name in metrics
        if name in baseline.metrics
    }


def _metric_rows(
    report: RunReport, baseline: Baseline | None
) -> list[tuple[str, str, str]]:
    diff = baseline_diff(report.metrics, baseline) if baseline else {}
    rows: list[tuple[str, str, str]] = []
    for name in sorted(report.metrics):
        delta = f"{diff[name]:+.4f}" if name in diff else ""
        rows.append((name, f"{report.metrics[name]:.4f}", delta))
    return rows


def failed_scores(report: RunReport) -> list[Score]:
    return [score for score in report.scores if score.passed is False]


def render_markdown(report: RunReport, baseline: Baseline | None = None) -> str:
    lines = [
        f"# evalgate run `{report.run_id}`",
        "",
        f"- dataset: `{report.dataset}`",
        f"- target: `{report.target}`",
        f"- cases: {len(report.results)}",
    ]
    if baseline is not None:
        lines.append(f"- baseline: `{baseline.run_id}`")
    lines += ["", "| metric | value | vs baseline |", "|---|---:|---:|"]
    lines += [
        f"| {name} | {value} | {delta} |"
        for name, value, delta in _metric_rows(report, baseline)
    ]
    failures = failed_scores(report)
    if failures:
        lines += ["", "## Failures", ""]
        lines += [
            f"- **{score.case_id}** `{score.scorer}`: {score.detail}"
            for score in failures
        ]
    return "\n".join(lines) + "\n"


def render_html(report: RunReport, baseline: Baseline | None = None) -> str:
    metric_rows = "\n".join(
        f"<tr><td>{escape(name)}</td><td>{value}</td><td>{delta}</td></tr>"
        for name, value, delta in _metric_rows(report, baseline)
    )
    failure_items = "\n".join(
        f"<li><b>{escape(score.case_id)}</b> <code>{escape(score.scorer)}</code>: "
        f"{escape(score.detail)}</li>"
        for score in failed_scores(report)
    ) or "<li>none</li>"
    return (
        f"<h1>evalgate run {escape(report.run_id)}</h1>\n"
        f"<p>dataset: <code>{escape(report.dataset)}</code> · "
        f"target: <code>{escape(report.target)}</code> · "
        f"cases: {len(report.results)}</p>\n"
        "<table><thead><tr><th>metric</th><th>value</th><th>vs baseline</th>"
        f"</tr></thead><tbody>\n{metric_rows}\n</tbody></table>\n"
        f"<h2>Failures</h2>\n<ul>\n{failure_items}\n</ul>\n"
    )
