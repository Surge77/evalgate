"""Adapter that wraps an in-process Python callable as an eval target."""

import importlib
import inspect
import time
from collections.abc import Awaitable, Callable

from evalgate.schemas import AgentResult, Case

TargetReturn = str | AgentResult
TargetFn = Callable[[Case], TargetReturn | Awaitable[TargetReturn]]


class CallableAdapter:
    """Runs a sync or async callable `(Case) -> str | AgentResult`.

    Exceptions become error results so one bad case never aborts a run.
    """

    def __init__(self, fn: TargetFn, name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    @classmethod
    def from_spec(cls, spec: str) -> "CallableAdapter":
        """Build from a `module.path:attribute` string (CLI entry point)."""
        module_path, _, attr = spec.partition(":")
        if not module_path or not attr:
            raise ValueError(f"invalid callable spec {spec!r}, expected 'module:attr'")
        module = importlib.import_module(module_path)
        fn: TargetFn = getattr(module, attr)
        return cls(fn, name=spec)

    async def run(self, case: Case) -> AgentResult:
        start = time.perf_counter()
        try:
            raw = self._fn(case)
            if inspect.isawaitable(raw):
                raw = await raw
        except Exception as exc:  # noqa: BLE001 — target failures are data, not crashes
            latency_ms = (time.perf_counter() - start) * 1000
            return AgentResult(
                case_id=case.id, output="", error=str(exc), latency_ms=latency_ms
            )
        latency_ms = (time.perf_counter() - start) * 1000

        if isinstance(raw, AgentResult):
            return raw.model_copy(
                update={
                    "case_id": case.id,
                    "latency_ms": raw.latency_ms or latency_ms,
                }
            )
        return AgentResult(case_id=case.id, output=raw, latency_ms=latency_ms)
