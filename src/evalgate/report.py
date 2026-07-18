"""Aggregate metrics from raw results and scores."""

from statistics import mean

from evalgate.schemas import AgentResult, Score

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
