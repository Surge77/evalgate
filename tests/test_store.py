from datetime import UTC, datetime
from pathlib import Path

from evalgate.schemas import RunReport
from evalgate.store import RunStore, load_baseline, promote_baseline


def make_report(run_id: str, started: str, accuracy: float = 1.0) -> RunReport:
    return RunReport(
        run_id=run_id,
        started_at=datetime.fromisoformat(started).replace(tzinfo=UTC),
        dataset="d.jsonl",
        target="stub",
        metrics={"accuracy": accuracy},
    )


def test_save_and_get_round_trip(tmp_path: Path) -> None:
    report = make_report("r1", "2026-07-18T10:00:00")
    with RunStore(tmp_path / "runs.db") as store:
        store.save(report)
        assert store.get("r1") == report


def test_get_missing_returns_none(tmp_path: Path) -> None:
    with RunStore(tmp_path / "runs.db") as store:
        assert store.get("absent") is None
        assert store.latest() is None


def test_latest_returns_newest_run(tmp_path: Path) -> None:
    with RunStore(tmp_path / "runs.db") as store:
        store.save(make_report("r1", "2026-07-18T10:00:00"))
        store.save(make_report("r2", "2026-07-18T11:00:00"))
        latest = store.latest()
    assert latest is not None and latest.run_id == "r2"


def test_list_run_ids_newest_first_with_limit(tmp_path: Path) -> None:
    with RunStore(tmp_path / "runs.db") as store:
        for hour in (10, 11, 12):
            store.save(make_report(f"r{hour}", f"2026-07-18T{hour}:00:00"))
        assert store.list_run_ids(limit=2) == ["r12", "r11"]


def test_save_same_run_id_overwrites(tmp_path: Path) -> None:
    with RunStore(tmp_path / "runs.db") as store:
        store.save(make_report("r1", "2026-07-18T10:00:00", accuracy=0.5))
        store.save(make_report("r1", "2026-07-18T10:00:00", accuracy=0.9))
        fetched = store.get("r1")
    assert fetched is not None and fetched.metrics["accuracy"] == 0.9


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    with RunStore(db) as store:
        store.save(make_report("r1", "2026-07-18T10:00:00"))
    with RunStore(db) as reopened:
        assert reopened.get("r1") is not None


def test_promote_and_load_baseline(tmp_path: Path) -> None:
    report = make_report("r1", "2026-07-18T10:00:00", accuracy=0.87)
    path = tmp_path / "baseline.json"
    promoted = promote_baseline(report, path)
    loaded = load_baseline(path)
    assert loaded == promoted
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.metrics == {"accuracy": 0.87}


def test_load_missing_baseline_returns_none(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "absent.json") is None
