"""Typer CLI: run (more commands land in later milestones)."""

import asyncio
import importlib
import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evalgate import __version__
from evalgate.adapters.callable import CallableAdapter
from evalgate.cache import ResponseCache
from evalgate.loader import DatasetError, load_dataset
from evalgate.providers import AnthropicProvider, Provider
from evalgate.report import (
    aggregate_metrics,
    baseline_diff,
    render_html,
    render_markdown,
)
from evalgate.runner import run_cases
from evalgate.schemas import Baseline, Case, RunReport
from evalgate.scorers.judge import judge_all
from evalgate.scorers.rules import score_case
from evalgate.store import RunStore, load_baseline, promote_baseline

DEFAULT_ARTIFACTS = Path("artifacts")
DEFAULT_BASELINE = Path("evalgate.baseline.json")

app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True, help="Baseline management")
app.add_typer(baseline_app, name="baseline")
console = Console()


class ReportFormat(str, Enum):
    RICH = "rich"
    MD = "md"
    HTML = "html"


@app.command()
def version() -> None:
    console.print(f"evalgate {__version__}")


@app.command()
def run(
    target: Annotated[
        str, typer.Option(help="Target callable as 'module.path:function'")
    ],
    dataset: Annotated[
        Path, typer.Option(help="Golden dataset JSONL")
    ] = Path("datasets/golden/sample.jsonl"),
    artifacts_dir: Annotated[
        Path, typer.Option(help="Where run reports and cache live")
    ] = DEFAULT_ARTIFACTS,
    concurrency: Annotated[int, typer.Option(min=1)] = 4,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Run only the first N cases")
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    judge_provider: Annotated[
        str | None,
        typer.Option(
            help="LLM judge: 'anthropic' or a 'module:attr' Provider spec"
        ),
    ] = None,
) -> None:
    """Run the dataset through a target and score it with rule checks."""
    try:
        cases = load_dataset(dataset)
    except DatasetError as exc:
        console.print(f"[red]dataset error:[/red] {exc}")
        raise typer.Exit(2) from exc
    if limit is not None:
        cases = cases[:limit]

    try:
        adapter = CallableAdapter.from_spec(target)
        judge = _load_judge_provider(judge_provider)
    except (ValueError, ImportError, AttributeError) as exc:
        console.print(f"[red]target error:[/red] {exc}")
        raise typer.Exit(2) from exc

    report = _execute(
        cases, adapter, judge, dataset, artifacts_dir, concurrency, no_cache
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / f"{report.run_id}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    with RunStore(artifacts_dir / "runs.db") as store:
        store.save(report)

    _print_summary(report)
    console.print(f"report written to [bold]{report_path}[/bold]")


def _load_run(artifacts_dir: Path, run_id: str | None) -> RunReport:
    with RunStore(artifacts_dir / "runs.db") as store:
        report = store.latest() if run_id is None else store.get(run_id)
    if report is None:
        which = run_id or "latest"
        console.print(f"[red]run not found:[/red] {which}")
        raise typer.Exit(2)
    return report


@app.command()
def report(
    run_id: Annotated[
        str | None, typer.Option(help="Run to render (default: latest)")
    ] = None,
    artifacts_dir: Annotated[Path, typer.Option()] = DEFAULT_ARTIFACTS,
    fmt: Annotated[ReportFormat, typer.Option("--format")] = ReportFormat.RICH,
    output: Annotated[
        Path | None, typer.Option(help="Write to file instead of stdout")
    ] = None,
    baseline_path: Annotated[Path, typer.Option("--baseline")] = DEFAULT_BASELINE,
) -> None:
    """Render a stored run as rich terminal output, markdown, or html."""
    run_report = _load_run(artifacts_dir, run_id)
    baseline = load_baseline(baseline_path)

    if fmt is ReportFormat.RICH:
        _print_summary(run_report, baseline)
        return
    rendered = (
        render_markdown(run_report, baseline)
        if fmt is ReportFormat.MD
        else render_html(run_report, baseline)
    )
    if output is None:
        console.print(rendered, markup=False)
    else:
        output.write_text(rendered, encoding="utf-8")
        console.print(f"report written to [bold]{output}[/bold]")


@baseline_app.command("promote")
def baseline_promote(
    run_id: Annotated[
        str | None, typer.Argument(help="Run to promote (default: latest)")
    ] = None,
    artifacts_dir: Annotated[Path, typer.Option()] = DEFAULT_ARTIFACTS,
    baseline_path: Annotated[Path, typer.Option("--baseline")] = DEFAULT_BASELINE,
) -> None:
    """Promote a run's metrics to the repo baseline JSON."""
    run_report = _load_run(artifacts_dir, run_id)
    baseline = promote_baseline(run_report, baseline_path)
    console.print(
        f"baseline [bold]{baseline.run_id}[/bold] written to {baseline_path}"
    )


def _load_judge_provider(spec: str | None) -> Provider | None:
    if spec is None:
        return None
    if spec == "anthropic":
        return AnthropicProvider()
    module_path, _, attr = spec.partition(":")
    if not module_path or not attr:
        raise ValueError(f"invalid judge provider spec {spec!r}, expected 'module:attr'")
    candidate: object = getattr(importlib.import_module(module_path), attr)
    if not isinstance(candidate, Provider):
        raise ValueError(f"judge provider {spec!r} does not implement Provider")
    return candidate


def _execute(
    cases: list[Case],
    adapter: CallableAdapter,
    judge: Provider | None,
    dataset: Path,
    artifacts_dir: Path,
    concurrency: int,
    no_cache: bool,
) -> RunReport:
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    cache = None if no_cache else ResponseCache(artifacts_dir / "cache.db")
    judge_cost = 0.0
    try:
        results = asyncio.run(
            run_cases(cases, adapter, cache=cache, concurrency=concurrency)
        )
        scores = [
            score for case, result in zip(cases, results, strict=True)
            for score in score_case(case, result)
        ]
        if judge is not None:
            judge_scores, judge_cost = asyncio.run(
                judge_all(cases, results, judge, cache=cache, concurrency=concurrency)
            )
            scores.extend(judge_scores)
    finally:
        if cache is not None:
            cache.close()

    metrics = aggregate_metrics(results, scores)
    if judge is not None:
        metrics["judge_cost_usd"] = judge_cost
        metrics["total_cost_usd"] = metrics.get("total_cost_usd", 0.0) + judge_cost
    return RunReport(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        dataset=str(dataset),
        target=adapter.name,
        results=results,
        scores=scores,
        metrics=metrics,
    )


def _print_summary(report: RunReport, baseline: Baseline | None = None) -> None:
    diff = baseline_diff(report.metrics, baseline) if baseline else {}
    table = Table(title=f"evalgate run {report.run_id}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    if baseline:
        table.add_column("vs baseline", justify="right")
    for name, value in sorted(report.metrics.items()):
        row = [name, f"{value:.4f}"]
        if baseline:
            row.append(f"{diff[name]:+.4f}" if name in diff else "")
        table.add_row(*row)
    console.print(table)
    failed = [score for score in report.scores if score.passed is False]
    for score in failed:
        console.print(f"[red]FAIL[/red] {score.case_id} {score.scorer}: {score.detail}")


if __name__ == "__main__":
    app()
