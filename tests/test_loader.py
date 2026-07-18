from pathlib import Path

import pytest

from evalgate.loader import DatasetError, load_dataset

SAMPLE = Path("datasets/golden/sample.jsonl")


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


VALID_LINE = (
    '{"id": "case-001", "category": "qa", "input": {"q": "x"},'
    ' "checks": [{"type": "exact", "value": "4"}]}'
)


def test_loads_repo_sample_dataset() -> None:
    cases = load_dataset(SAMPLE)
    assert [case.id for case in cases] == ["case-001", "case-002"]
    assert cases[0].checks[0].type.value == "contains"


def test_golden_v1_has_60_cases_across_5_categories() -> None:
    cases = load_dataset(Path("datasets/golden/golden_v1.jsonl"))
    assert len(cases) == 60
    categories = {case.category for case in cases}
    assert categories == {"qa", "extraction", "math", "classification", "tool_plan"}
    assert all(case.checks for case in cases), "every golden case needs a rule check"


def test_skips_blank_lines(tmp_path: Path) -> None:
    dataset = write_lines(tmp_path / "d.jsonl", [VALID_LINE, "", "   "])
    assert len(load_dataset(dataset)) == 1


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    dataset = write_lines(tmp_path / "d.jsonl", [VALID_LINE, "{not json"])
    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(dataset)


def test_validation_failure_reports_line_number(tmp_path: Path) -> None:
    bad = '{"id": "", "category": "qa", "input": {}}'
    dataset = write_lines(tmp_path / "d.jsonl", [bad])
    with pytest.raises(DatasetError, match="line 1"):
        load_dataset(dataset)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    dataset = write_lines(tmp_path / "d.jsonl", [VALID_LINE, VALID_LINE])
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(dataset)


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    future = (
        '{"schema_version": 99, "id": "c", "category": "qa", "input": {}}'
    )
    dataset = write_lines(tmp_path / "d.jsonl", [future])
    with pytest.raises(DatasetError, match="schema_version"):
        load_dataset(dataset)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_empty_dataset_rejected(tmp_path: Path) -> None:
    dataset = write_lines(tmp_path / "d.jsonl", [""])
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(dataset)
