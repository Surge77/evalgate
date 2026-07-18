import asyncio
from pathlib import Path

from evalgate.cache import ResponseCache
from evalgate.runner import run_cases
from evalgate.schemas import AgentResult, Case


def make_cases(count: int) -> list[Case]:
    return [
        Case.model_validate({"id": f"c{i}", "category": "qa", "input": {"q": str(i)}})
        for i in range(count)
    ]


class CountingAdapter:
    """Echoes case ids; optionally fails the first N calls per case."""

    name = "counting"

    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[str] = []
        self._failures_left: dict[str, int] = {}
        self._fail_first = fail_first

    async def run(self, case: Case) -> AgentResult:
        self.calls.append(case.id)
        left = self._failures_left.setdefault(case.id, self._fail_first)
        if left > 0:
            self._failures_left[case.id] = left - 1
            return AgentResult(case_id=case.id, output="", error="flaky")
        return AgentResult(case_id=case.id, output=f"answer-{case.id}")


async def test_results_keep_dataset_order() -> None:
    cases = make_cases(5)
    results = await run_cases(cases, CountingAdapter(), concurrency=3)
    assert [result.case_id for result in results] == [case.id for case in cases]


async def test_concurrency_is_bounded() -> None:
    active = 0
    peak = 0

    class SlowAdapter:
        name = "slow"

        async def run(self, case: Case) -> AgentResult:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return AgentResult(case_id=case.id, output="ok")

    await run_cases(make_cases(8), SlowAdapter(), concurrency=2)
    assert peak <= 2


async def test_retries_flaky_target() -> None:
    adapter = CountingAdapter(fail_first=1)
    results = await run_cases(
        make_cases(1), adapter, retries=1, backoff_s=0.0
    )
    assert results[0].error is None
    assert adapter.calls == ["c0", "c0"]


async def test_no_retries_returns_error_result() -> None:
    adapter = CountingAdapter(fail_first=1)
    results = await run_cases(make_cases(1), adapter, retries=0)
    assert results[0].error == "flaky"


async def test_cache_hit_skips_adapter(tmp_path: Path) -> None:
    cases = make_cases(3)
    adapter = CountingAdapter()
    with ResponseCache(tmp_path / "cache.db") as cache:
        await run_cases(cases, adapter, cache=cache)
        first_calls = len(adapter.calls)
        results = await run_cases(cases, adapter, cache=cache)
    assert first_calls == 3
    assert len(adapter.calls) == 3, "second run must be fully cached"
    assert [result.output for result in results] == [
        "answer-c0",
        "answer-c1",
        "answer-c2",
    ]


async def test_error_results_are_not_cached(tmp_path: Path) -> None:
    adapter = CountingAdapter(fail_first=1)
    with ResponseCache(tmp_path / "cache.db") as cache:
        await run_cases(make_cases(1), adapter, cache=cache, retries=0)
        results = await run_cases(make_cases(1), adapter, cache=cache, retries=0)
    assert results[0].error is None, "second attempt should re-call and succeed"
    assert adapter.calls == ["c0", "c0"]


async def test_case_content_change_invalidates_cache(tmp_path: Path) -> None:
    adapter = CountingAdapter()
    original = make_cases(1)
    edited = [
        original[0].model_copy(update={"input": {"q": "reworded question"}})
    ]
    with ResponseCache(tmp_path / "cache.db") as cache:
        await run_cases(original, adapter, cache=cache)
        await run_cases(edited, adapter, cache=cache)
    assert len(adapter.calls) == 2
