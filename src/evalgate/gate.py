"""CI regression gate: threshold rules → pass/fail with GitHub-friendly output."""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalgate.schemas import Baseline

PERCENT = 100.0


class GateError(Exception):
    """Raised for a missing or invalid gate configuration."""


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: Path = Path("evalgate.baseline.json")
    max_drop: dict[str, float] = Field(default_factory=dict)
    min: dict[str, float] = Field(default_factory=dict)
    warn_increase_pct: dict[str, float] = Field(default_factory=dict)


class GateOutcome(BaseModel):
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def load_gate_config(path: Path) -> GateConfig:
    if not path.is_file():
        raise GateError(f"gate config not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise GateError(f"invalid TOML in {path}: {exc}") from exc
    section = data.get("gate")
    if not isinstance(section, dict):
        raise GateError(f"{path} has no [gate] section")
    try:
        return GateConfig.model_validate(section)
    except ValidationError as exc:
        raise GateError(f"invalid [gate] config in {path}: {exc}") from exc


def evaluate_gate(
    metrics: dict[str, float], baseline: Baseline | None, config: GateConfig
) -> GateOutcome:
    outcome = GateOutcome()
    _check_floors(metrics, config, outcome)
    _check_baseline_rules(metrics, baseline, config, outcome)
    return outcome


def _check_floors(
    metrics: dict[str, float], config: GateConfig, outcome: GateOutcome
) -> None:
    for name, floor in config.min.items():
        value = metrics.get(name)
        if value is None:
            outcome.failures.append(f"metric {name!r} missing from run (min={floor})")
        elif value < floor:
            outcome.failures.append(f"{name} {value:.4f} below minimum {floor:.4f}")


def _check_baseline_rules(
    metrics: dict[str, float],
    baseline: Baseline | None,
    config: GateConfig,
    outcome: GateOutcome,
) -> None:
    if not config.max_drop and not config.warn_increase_pct:
        return
    if baseline is None:
        outcome.failures.append(
            "no baseline promoted — run 'evalgate baseline promote' first"
        )
        return

    for name, allowed_drop in config.max_drop.items():
        value, base = metrics.get(name), baseline.metrics.get(name)
        if value is None or base is None:
            outcome.failures.append(
                f"metric {name!r} missing from run or baseline (max_drop={allowed_drop})"
            )
            continue
        drop = base - value
        if drop > allowed_drop:
            outcome.failures.append(
                f"{name} dropped {drop:.4f} vs baseline {base:.4f} "
                f"(allowed {allowed_drop:.4f})"
            )

    for name, allowed_pct in config.warn_increase_pct.items():
        value, base = metrics.get(name), baseline.metrics.get(name)
        if value is None or base is None or base == 0:
            continue
        increase_pct = (value - base) / base * PERCENT
        if increase_pct > allowed_pct:
            outcome.warnings.append(
                f"{name} rose {increase_pct:.1f}% vs baseline (warn above {allowed_pct:.1f}%)"
            )


def format_for_ci(outcome: GateOutcome) -> list[str]:
    """Annotation lines; ::error/::warning form under GitHub Actions."""
    on_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    lines = [
        f"::error::evalgate: {failure}" if on_actions else f"FAIL {failure}"
        for failure in outcome.failures
    ]
    lines += [
        f"::warning::evalgate: {warning}" if on_actions else f"WARN {warning}"
        for warning in outcome.warnings
    ]
    return lines
