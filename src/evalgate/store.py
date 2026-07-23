"""SQLite run history and baseline promotion."""

import sqlite3
from pathlib import Path
from types import TracebackType

from evalgate.schemas import Baseline, RunReport


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            " run_id TEXT PRIMARY KEY,"
            " started_at TEXT NOT NULL,"
            " report TEXT NOT NULL)"
        )
        self._conn.commit()

    def save(self, report: RunReport) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, report) VALUES (?, ?, ?)",
            (report.run_id, report.started_at.isoformat(), report.model_dump_json()),
        )
        self._conn.commit()

    def get(self, run_id: str) -> RunReport | None:
        row = self._conn.execute(
            "SELECT report FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else RunReport.model_validate_json(str(row[0]))

    def latest(self) -> RunReport | None:
        row = self._conn.execute(
            "SELECT report FROM runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else RunReport.model_validate_json(str(row[0]))

    def list_run_ids(self, limit: int = 20) -> list[str]:
        rows = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC, run_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def promote_baseline(report: RunReport, baseline_path: Path) -> Baseline:
    """Write the run's metrics as the repo baseline JSON. Returns the baseline."""
    baseline = Baseline(run_id=report.run_id, metrics=report.metrics)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    return baseline


def load_baseline(baseline_path: Path) -> Baseline | None:
    if not baseline_path.is_file():
        return None
    return Baseline.model_validate_json(baseline_path.read_text(encoding="utf-8"))
