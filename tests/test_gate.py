from pathlib import Path

import pytest

from evalgate.gate import (
    GateConfig,
    GateError,
    evaluate_gate,
    format_for_ci,
    load_gate_config,
)
from evalgate.schemas import Baseline

BASELINE = Baseline(
    run_id="r0",
    metrics={"accuracy": 0.90, "faithfulness": 0.85, "total_cost_usd": 1.00},
)


def config(**overrides: object) -> GateConfig:
    return GateConfig.model_validate(overrides)


class TestMaxDrop:
    def test_passes_within_allowed_drop(self) -> None:
        cfg = config(max_drop={"accuracy": 0.05})
        outcome = evaluate_gate({"accuracy": 0.88}, BASELINE, cfg)
        assert outcome.passed

    def test_fails_beyond_allowed_drop(self) -> None:
        cfg = config(max_drop={"accuracy": 0.05})
        outcome = evaluate_gate({"accuracy": 0.80}, BASELINE, cfg)
        assert not outcome.passed
        assert "accuracy dropped" in outcome.failures[0]

    def test_improvement_always_passes(self) -> None:
        cfg = config(max_drop={"accuracy": 0.0})
        assert evaluate_gate({"accuracy": 0.95}, BASELINE, cfg).passed

    def test_missing_metric_in_run_fails(self) -> None:
        cfg = config(max_drop={"faithfulness": 0.05})
        outcome = evaluate_gate({"accuracy": 0.9}, BASELINE, cfg)
        assert not outcome.passed
        assert "missing" in outcome.failures[0]

    def test_missing_metric_in_baseline_fails(self) -> None:
        cfg = config(max_drop={"brand_new_metric": 0.05})
        outcome = evaluate_gate({"brand_new_metric": 0.9}, BASELINE, cfg)
        assert not outcome.passed

    def test_no_baseline_fails_when_drop_rules_exist(self) -> None:
        cfg = config(max_drop={"accuracy": 0.05})
        outcome = evaluate_gate({"accuracy": 0.99}, None, cfg)
        assert not outcome.passed
        assert "no baseline" in outcome.failures[0]


class TestMinFloor:
    def test_floor_pass_and_fail(self) -> None:
        cfg = config(min={"accuracy": 0.8})
        assert evaluate_gate({"accuracy": 0.8}, None, cfg).passed
        assert not evaluate_gate({"accuracy": 0.79}, None, cfg).passed

    def test_floor_works_without_baseline(self) -> None:
        cfg = config(min={"accuracy": 0.8})
        assert evaluate_gate({"accuracy": 0.9}, None, cfg).passed

    def test_missing_metric_fails_floor(self) -> None:
        cfg = config(min={"accuracy": 0.8})
        assert not evaluate_gate({}, None, cfg).passed


class TestWarnIncrease:
    def test_warns_but_passes_on_cost_rise(self) -> None:
        cfg = config(warn_increase_pct={"total_cost_usd": 20.0})
        outcome = evaluate_gate({"total_cost_usd": 1.50}, BASELINE, cfg)
        assert outcome.passed
        assert "rose 50.0%" in outcome.warnings[0]

    def test_no_warning_within_threshold(self) -> None:
        cfg = config(warn_increase_pct={"total_cost_usd": 20.0})
        outcome = evaluate_gate({"total_cost_usd": 1.10}, BASELINE, cfg)
        assert outcome.warnings == []

    def test_zero_baseline_cost_never_warns(self) -> None:
        baseline = Baseline(run_id="r0", metrics={"total_cost_usd": 0.0})
        cfg = config(warn_increase_pct={"total_cost_usd": 20.0})
        assert evaluate_gate({"total_cost_usd": 5.0}, baseline, cfg).warnings == []


def test_empty_config_always_passes() -> None:
    assert evaluate_gate({}, None, config()).passed


class TestLoadConfig:
    def test_loads_full_config(self, tmp_path: Path) -> None:
        path = tmp_path / "evalgate.toml"
        path.write_text(
            '[gate]\nbaseline = "b.json"\n'
            "[gate.max_drop]\naccuracy = 0.05\n"
            "[gate.min]\naccuracy = 0.8\n"
            "[gate.warn_increase_pct]\ntotal_cost_usd = 20.0\n",
            encoding="utf-8",
        )
        cfg = load_gate_config(path)
        assert cfg.baseline == Path("b.json")
        assert cfg.max_drop == {"accuracy": 0.05}
        assert cfg.min == {"accuracy": 0.8}
        assert cfg.warn_increase_pct == {"total_cost_usd": 20.0}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GateError, match="not found"):
            load_gate_config(tmp_path / "absent.toml")

    def test_invalid_toml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "evalgate.toml"
        path.write_text("[gate\nbroken", encoding="utf-8")
        with pytest.raises(GateError, match="invalid TOML"):
            load_gate_config(path)

    def test_missing_gate_section_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "evalgate.toml"
        path.write_text("[other]\nx = 1\n", encoding="utf-8")
        with pytest.raises(GateError, match="no \\[gate\\] section"):
            load_gate_config(path)

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "evalgate.toml"
        path.write_text("[gate]\ntypo_key = 1\n", encoding="utf-8")
        with pytest.raises(GateError, match="invalid \\[gate\\] config"):
            load_gate_config(path)


class TestCiFormatting:
    def test_plain_format_off_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        cfg = config(min={"accuracy": 0.8}, warn_increase_pct={"total_cost_usd": 1.0})
        outcome = evaluate_gate({"accuracy": 0.5, "total_cost_usd": 9.0}, BASELINE, cfg)
        lines = format_for_ci(outcome)
        assert lines[0].startswith("FAIL ")
        assert lines[-1].startswith("WARN ")

    def test_github_annotations_on_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        cfg = config(min={"accuracy": 0.8})
        outcome = evaluate_gate({"accuracy": 0.5}, None, cfg)
        assert format_for_ci(outcome)[0].startswith("::error::evalgate:")
