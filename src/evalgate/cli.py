"""Typer CLI: run, report, baseline promote, gate."""

import asyncio
import importlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from evalgate import __version__, cli_report
from evalgate.adapters.base import Adapter
from evalgate.adapters.callable import CallableAdapter
from evalgate.adapters.http import HttpAdapter
from evalgate.cache import ResponseCache
from evalgate.cli_report import DEFAULT_ARTIFACTS, console
from evalgate.loader import DatasetError, load_dataset
from evalgate.providers import (
    AnthropicProvider,
    ClaudeCliProvider,
    GeminiProvider,
    GroqProvider,
    Provider,
)
from evalgate.report import aggregate_metrics
from evalgate.runner import run_cases
from evalgate.schemas import Case, RunReport
from evalgate.scorers.judge import judge_all
from evalgate.scorers.rules import score_case
from evalgate.store import RunStore

load_dotenv()  # ANTHROPIC_API_KEY for the live judge

app = typer.Typer(no_args_is_help=True)
app.command()(cli_report.report)
app.command()(cli_report.gate)
app.command()(cli_report.calibrate)
app.add_typer(cli_report.baseline_app, name="baseline")


@app.command()
def version() -> None:
    console.print(f"evalgate {__version__}")


@app.command()
def run(
    target: Annotated[
        str,
        typer.Option(help="Target: 'module.path:function' or an http(s) URL"),
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
            help="LLM judge: 'anthropic', 'claude-cli' (uses Claude Code auth, no API key), "
            "'gemini' (free-tier, GEMINI_API_KEY), 'groq' (free-tier, GROQ_API_KEY), "
            "or a 'module:attr' Provider spec"
        ),
    ] = None,
) -> None:
    """Run the dataset through a target and score it."""
    try:
        cases = load_dataset(dataset)
    except DatasetError as exc:
        console.print(f"[red]dataset error:[/red] {exc}")
        raise typer.Exit(2) from exc
    if limit is not None:
        cases = cases[:limit]

    try:
        adapter = _build_adapter(target)
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

    cli_report.print_summary(report)
    console.print(f"report written to [bold]{report_path}[/bold]")


def _build_adapter(target: str) -> Adapter:
    if target.startswith(("http://", "https://")):
        return HttpAdapter(target)
    return CallableAdapter.from_spec(target)


def _load_judge_provider(spec: str | None) -> Provider | None:
    if spec is None:
        return None
    if spec == "anthropic":
        return AnthropicProvider()
    if spec == "claude-cli":
        return ClaudeCliProvider()
    if spec == "gemini":
        return GeminiProvider()
    if spec == "groq":
        return GroqProvider()
    module_path, _, attr = spec.partition(":")
    if not module_path or not attr:
        raise ValueError(f"invalid judge provider spec {spec!r}, expected 'module:attr'")
    candidate: object = getattr(importlib.import_module(module_path), attr)
    if not isinstance(candidate, Provider):
        raise ValueError(f"judge provider {spec!r} does not implement Provider")
    return candidate


def _execute(
    cases: list[Case],
    adapter: Adapter,
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


if __name__ == "__main__":
    app()
