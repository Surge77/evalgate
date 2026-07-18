import json
from pathlib import Path

from typer.testing import CliRunner

from evalgate.cli import app

runner = CliRunner()
SAMPLE = "datasets/golden/sample.jsonl"


def run_cli(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_version_command() -> None:
    exit_code, output = run_cli("version")
    assert exit_code == 0
    assert "evalgate" in output


def test_run_perfect_target_writes_report(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:golden_parrot",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
    )
    assert exit_code == 0, output
    reports = [p for p in tmp_path.glob("*.json")]
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["error_rate"] == 0.0
    assert len(report["results"]) == 2


def test_run_failing_target_reports_failures(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:echo_question",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
        "--no-cache",
    )
    assert exit_code == 0
    assert "FAIL" in output


def test_run_limit_restricts_cases(tmp_path: Path) -> None:
    exit_code, _ = run_cli(
        "run",
        "--target", "tests.targets:golden_parrot",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
        "--limit", "1",
        "--no-cache",
    )
    assert exit_code == 0
    report = json.loads(
        next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    )
    assert len(report["results"]) == 1


def test_run_missing_dataset_exits_2(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:golden_parrot",
        "--dataset", str(tmp_path / "nope.jsonl"),
        "--artifacts-dir", str(tmp_path),
    )
    assert exit_code == 2
    assert "dataset error" in output


def test_run_bad_target_spec_exits_2(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "not-a-spec",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
    )
    assert exit_code == 2
    assert "target error" in output


def test_run_with_stub_judge_adds_judge_metrics(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:golden_parrot",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
        "--judge-provider", "tests.judge_stub:provider",
        "--no-cache",
    )
    assert exit_code == 0, output
    report = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert report["metrics"]["faithfulness"] == 1.0
    assert report["metrics"]["correctness"] == 1.0
    assert report["metrics"]["relevance"] == 1.0
    assert report["metrics"]["judge_cost_usd"] > 0
    judge_scores = [
        score for score in report["scores"] if score["scorer"].startswith("judge:")
    ]
    assert len(judge_scores) == 6


def test_run_bad_judge_spec_exits_2(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:golden_parrot",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
        "--judge-provider", "tests.targets:echo_question",
    )
    assert exit_code == 2
    assert "target error" in output


def test_run_uses_cache_between_invocations(tmp_path: Path) -> None:
    for _ in range(2):
        exit_code, _ = run_cli(
            "run",
            "--target", "tests.targets:golden_parrot",
            "--dataset", SAMPLE,
            "--artifacts-dir", str(tmp_path),
        )
        assert exit_code == 0
    assert (tmp_path / "cache.db").exists()
