"""Core Pydantic models shared by every evalgate module."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CASE_SCHEMA_VERSION = 1


class CheckType(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    TOOL_SEQUENCE = "tool_sequence"


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: CheckType
    value: str
    tolerance: float | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = CASE_SCHEMA_VERSION
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    input: dict[str, Any]
    context: list[str] = Field(default_factory=list)
    reference: str | None = None
    checks: list[Check] = Field(default_factory=list[Check])
    tags: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    output: str
    tool_calls: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    scorer: str
    value: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    detail: str = ""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    dataset: str
    target: str
    results: list[AgentResult] = Field(default_factory=list[AgentResult])
    scores: list[Score] = Field(default_factory=list[Score])
    metrics: dict[str, float] = Field(default_factory=dict)


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    promoted_at: datetime = Field(default_factory=_utcnow)
    metrics: dict[str, float] = Field(default_factory=dict)
