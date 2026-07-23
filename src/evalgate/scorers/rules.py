"""Rule-based scorers. Pure functions — no I/O, no state."""

import json
import re
from collections.abc import Callable
from typing import Any

from evalgate.schemas import AgentResult, Case, Check, CheckType, Score

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def score_case(case: Case, result: AgentResult) -> list[Score]:
    return [_score_check(case, check, result) for check in case.checks]


def _score_check(case: Case, check: Check, result: AgentResult) -> Score:
    scorer = f"rule:{check.type.value}"
    if result.error is not None:
        return _fail(case, scorer, f"target error: {result.error}")

    passed, detail = _CHECKS[check.type](check, result)
    if passed:
        return Score(case_id=case.id, scorer=scorer, value=1.0, passed=True)
    return _fail(case, scorer, detail)


def _fail(case: Case, scorer: str, detail: str) -> Score:
    return Score(case_id=case.id, scorer=scorer, value=0.0, passed=False, detail=detail)


def _check_exact(check: Check, result: AgentResult) -> tuple[bool, str]:
    if result.output.strip() == check.value.strip():
        return True, ""
    return False, f"expected exact {check.value!r}, got {result.output!r}"


def _check_contains(check: Check, result: AgentResult) -> tuple[bool, str]:
    if check.value in result.output:
        return True, ""
    return False, f"output does not contain {check.value!r}"


def _check_regex(check: Check, result: AgentResult) -> tuple[bool, str]:
    if re.search(check.value, result.output):
        return True, ""
    return False, f"output does not match /{check.value}/"


def _check_numeric_tolerance(check: Check, result: AgentResult) -> tuple[bool, str]:
    expected = float(check.value)
    tolerance = check.tolerance or 0.0
    numbers = [float(match) for match in _NUMBER_RE.findall(result.output)]
    if not numbers:
        return False, "no number found in output"
    # Free-text answers repeat operands ("480 / 12 is 40"), so any number
    # within tolerance passes — position in the sentence is not meaningful.
    if any(abs(actual - expected) <= tolerance for actual in numbers):
        return True, ""
    return False, f"expected {expected} ± {tolerance}, got {numbers}"


def _check_tool_sequence(check: Check, result: AgentResult) -> tuple[bool, str]:
    expected = [name.strip() for name in check.value.split(",") if name.strip()]
    if result.tool_calls == expected:
        return True, ""
    return False, f"expected tools {expected}, got {result.tool_calls}"


def _check_json_schema(check: Check, result: AgentResult) -> tuple[bool, str]:
    try:
        schema = json.loads(check.value)
    except json.JSONDecodeError as exc:
        return False, f"invalid schema in check: {exc}"
    try:
        data = json.loads(result.output)
    except json.JSONDecodeError:
        return False, "output is not valid JSON"
    error = _validate_schema(data, schema, path="$")
    if error is None:
        return True, ""
    return False, error


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _validate_schema(data: Any, schema: dict[str, Any], path: str) -> str | None:
    """Minimal JSON-Schema subset: type, required, properties. Returns error or None."""
    expected_type = schema.get("type")
    if expected_type is not None:
        py_type = _JSON_TYPES.get(expected_type)
        if py_type is None:
            return f"{path}: unsupported schema type {expected_type!r}"
        if isinstance(data, bool) and expected_type in ("number", "integer"):
            return f"{path}: expected {expected_type}, got boolean"
        if not isinstance(data, py_type):
            return f"{path}: expected {expected_type}, got {type(data).__name__}"

    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                return f"{path}: missing required key {key!r}"
        properties: dict[str, Any] = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in data:
                error = _validate_schema(data[key], sub_schema, f"{path}.{key}")
                if error is not None:
                    return error
    return None


_CHECKS: dict[CheckType, Callable[[Check, AgentResult], tuple[bool, str]]] = {
    CheckType.EXACT: _check_exact,
    CheckType.CONTAINS: _check_contains,
    CheckType.REGEX: _check_regex,
    CheckType.JSON_SCHEMA: _check_json_schema,
    CheckType.NUMERIC_TOLERANCE: _check_numeric_tolerance,
    CheckType.TOOL_SEQUENCE: _check_tool_sequence,
}
