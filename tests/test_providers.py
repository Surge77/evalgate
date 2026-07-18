from evalgate.providers import JUDGE_MODEL, PRICING_PER_MTOK, compute_cost_usd


def test_judge_model_has_pricing() -> None:
    assert JUDGE_MODEL in PRICING_PER_MTOK


def test_cost_computed_from_pricing_table() -> None:
    cost = compute_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
    assert cost == 5.00 + 25.00


def test_cost_scales_with_tokens() -> None:
    assert compute_cost_usd("claude-haiku-4-5", 500_000, 0) == 0.50


def test_unknown_model_costs_zero() -> None:
    assert compute_cost_usd("some-future-model", 1_000_000, 1_000_000) == 0.0
