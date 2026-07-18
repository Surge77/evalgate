"""CLI commands for inspecting runs: report, baseline promote, gate."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evalgate.gate import GateError, evaluate_gate, format_for_ci, load_gate_config
from evalgate.report import baseline_diff, render_html, render_markdown
from evalgate.schemas import Baseline, RunReport
from evalgate.store import RunStore, load_baseline, promote_baseline

DEFAULT_ARTIFACTS = Path("artifacts")
DEFAULT_BASELINE = Path("evalgate.baseline.json")
DEFAULT_GATE_CONFIG = Path("evalgate.toml")

console = Console()
baseline_app = typer.Typer(no_args_is_help=True, help="Baseline management")


class ReportFormat(str, Enum):
    RICH = "rich"
    MD = "md"
    HTML = "html"


def print_summary(report: RunReport, baseline: Baseline | None = None) -> None:
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
    for score in report.scores:
        if score.passed is False:
            console.print(
                f"[red]FAIL[/red] {score.case_id} {score.scorer}: {score.detail}"
            )


def load_run(artifacts_dir: Path, run_id: str | None) -> RunReport:
    with RunStore(artifacts_dir / "runs.db") as store:
        report = store.latest() if run_id is None else store.get(run_id)
    if report is None:
        console.print(f"[red]run not found:[/red] {run_id or 'latest'}")
        raise typer.Exit(2)
    return report


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
    run_report = load_run(artifacts_dir, run_id)
    baseline = load_baseline(baseline_path)

    if fmt is ReportFormat.RICH:
        print_summary(run_report, baseline)
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
    run_report = load_run(artifacts_dir, run_id)
    baseline = promote_baseline(run_report, baseline_path)
    console.print(
        f"baseline [bold]{baseline.run_id}[/bold] written to {baseline_path}"
    )


def gate(
    run_id: Annotated[
        str | None, typer.Option(help="Run to check (default: latest)")
    ] = None,
    artifacts_dir: Annotated[Path, typer.Option()] = DEFAULT_ARTIFACTS,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_GATE_CONFIG,
) -> None:
    """Compare a run against the baseline thresholds; exit 1 on regression."""
    try:
        config = load_gate_config(config_path)
    except GateError as exc:
        console.print(f"[red]gate config error:[/red] {exc}")
        raise typer.Exit(2) from exc

    run_report = load_run(artifacts_dir, run_id)
    baseline = load_baseline(config.baseline)
    outcome = evaluate_gate(run_report.metrics, baseline, config)

    for line in format_for_ci(outcome):
        console.print(line, markup=False)
    if outcome.passed:
        console.print(f"[green]gate passed[/green] for run {run_report.run_id}")
        return
    console.print(f"[red]gate failed[/red] for run {run_report.run_id}")
    raise typer.Exit(1)
