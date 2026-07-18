"""JSONL golden-dataset loading and validation."""

import json
from pathlib import Path

from pydantic import ValidationError

from evalgate.schemas import CASE_SCHEMA_VERSION, Case


class DatasetError(Exception):
    """Raised when a dataset file is missing, malformed, or inconsistent."""


def load_dataset(path: Path) -> list[Case]:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")

    cases: list[Case] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        cases.append(_parse_case(line, path, line_no, seen_ids))

    if not cases:
        raise DatasetError(f"no cases in dataset: {path}")
    return cases


def _parse_case(
    line: str, path: Path, line_no: int, seen_ids: set[str]
) -> Case:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path.name} line {line_no}: invalid JSON — {exc}") from exc

    try:
        case = Case.model_validate(data)
    except ValidationError as exc:
        raise DatasetError(f"{path.name} line {line_no}: {exc}") from exc

    if case.schema_version != CASE_SCHEMA_VERSION:
        raise DatasetError(
            f"{path.name} line {line_no}: unsupported schema_version "
            f"{case.schema_version} (supported: {CASE_SCHEMA_VERSION})"
        )
    if case.id in seen_ids:
        raise DatasetError(f"{path.name} line {line_no}: duplicate case id {case.id!r}")
    seen_ids.add(case.id)
    return case
