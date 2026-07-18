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


def do_run(tmp_path: Path) -> None:
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:golden_parrot",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
        "--no-cache",
    )
    assert exit_code == 0, output


def test_report_renders_latest_run_rich(tmp_path: Path) -> None:
    do_run(tmp_path)
    exit_code, output = run_cli(
        "report", "--artifacts-dir", str(tmp_path),
        "--baseline", str(tmp_path / "no-baseline.json"),
    )
    assert exit_code == 0
    assert "accuracy" in output


def test_report_markdown_written_to_file(tmp_path: Path) -> None:
    do_run(tmp_path)
    out = tmp_path / "report.md"
    exit_code, _ = run_cli(
        "report", "--artifacts-dir", str(tmp_path),
        "--format", "md", "--output", str(out),
        "--baseline", str(tmp_path / "no-baseline.json"),
    )
    assert exit_code == 0
    assert "| accuracy | 1.0000 |" in out.read_text(encoding="utf-8")


def test_report_unknown_run_exits_2(tmp_path: Path) -> None:
    do_run(tmp_path)
    exit_code, output = run_cli(
        "report", "--artifacts-dir", str(tmp_path), "--run-id", "nope",
    )
    assert exit_code == 2
    assert "run not found" in output


def write_gate_config(tmp_path: Path) -> tuple[Path, Path]:
    baseline_path = tmp_path / "baseline.json"
    config_path = tmp_path / "evalgate.toml"
    config_path.write_text(
        f'[gate]\nbaseline = {json.dumps(str(baseline_path))}\n'
        "[gate.max_drop]\naccuracy = 0.05\n",
        encoding="utf-8",
    )
    return config_path, baseline_path


def test_gate_passes_after_promoting_good_run(tmp_path: Path) -> None:
    do_run(tmp_path)
    config_path, baseline_path = write_gate_config(tmp_path)
    exit_code, output = run_cli(
        "baseline", "promote",
        "--artifacts-dir", str(tmp_path), "--baseline", str(baseline_path),
    )
    assert exit_code == 0, output
    exit_code, output = run_cli(
        "gate", "--artifacts-dir", str(tmp_path), "--config", str(config_path),
    )
    assert exit_code == 0
    assert "gate passed" in output


def test_gate_blocks_seeded_regression(tmp_path: Path) -> None:
    do_run(tmp_path)  # golden_parrot: accuracy 1.0 → baseline
    config_path, baseline_path = write_gate_config(tmp_path)
    run_cli(
        "baseline", "promote",
        "--artifacts-dir", str(tmp_path), "--baseline", str(baseline_path),
    )
    # sabotaged target: echoes the question, fails every check
    exit_code, output = run_cli(
        "run",
        "--target", "tests.targets:echo_question",
        "--dataset", SAMPLE,
        "--artifacts-dir", str(tmp_path),
        "--no-cache",
    )
    assert exit_code == 0
    exit_code, output = run_cli(
        "gate", "--artifacts-dir", str(tmp_path), "--config", str(config_path),
    )
    assert exit_code == 1
    assert "gate failed" in output
    assert "accuracy dropped" in output


def test_gate_fails_without_baseline(tmp_path: Path) -> None:
    do_run(tmp_path)
    config_path, _ = write_gate_config(tmp_path)
    exit_code, output = run_cli(
        "gate", "--artifacts-dir", str(tmp_path), "--config", str(config_path),
    )
    assert exit_code == 1
    assert "no baseline" in output


def test_gate_missing_config_exits_2(tmp_path: Path) -> None:
    do_run(tmp_path)
    exit_code, output = run_cli(
        "gate", "--artifacts-dir", str(tmp_path),
        "--config", str(tmp_path / "absent.toml"),
    )
    assert exit_code == 2
    assert "gate config error" in output


def test_baseline_promote_then_report_shows_delta(tmp_path: Path) -> None:
    do_run(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    exit_code, output = run_cli(
        "baseline", "promote",
        "--artifacts-dir", str(tmp_path), "--baseline", str(baseline_path),
    )
    assert exit_code == 0, output
    assert baseline_path.exists()

    exit_code, output = run_cli(
        "report", "--artifacts-dir", str(tmp_path),
        "--format", "md", "--baseline", str(baseline_path),
    )
    assert exit_code == 0
    assert "+0.0000" in output
