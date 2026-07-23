"""Async orchestration: bounded concurrency, retries with backoff, cache."""

import asyncio

from evalgate.adapters.base import Adapter
from evalgate.cache import ResponseCache, cache_key
from evalgate.schemas import AgentResult, Case


async def run_cases(
    cases: list[Case],
    adapter: Adapter,
    cache: ResponseCache | None = None,
    concurrency: int = 4,
    retries: int = 1,
    backoff_s: float = 0.5,
) -> list[AgentResult]:
    """Run every case through the adapter. Results keep dataset order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case: Case) -> AgentResult:
        async with semaphore:
            return await _run_one(case, adapter, cache, retries, backoff_s)

    return list(await asyncio.gather(*(bounded(case) for case in cases)))


async def _run_one(
    case: Case,
    adapter: Adapter,
    cache: ResponseCache | None,
    retries: int,
    backoff_s: float,
) -> AgentResult:
    key = cache_key("target", adapter.name, case.model_dump_json())
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return AgentResult.model_validate_json(hit)

    result = await _run_with_retries(case, adapter, retries, backoff_s)
    if cache is not None and result.error is None:
        cache.put(key, result.model_dump_json())
    return result


async def _run_with_retries(
    case: Case, adapter: Adapter, retries: int, backoff_s: float
) -> AgentResult:
    result = await adapter.run(case)
    for attempt in range(retries):
        if result.error is None:
            break
        await asyncio.sleep(backoff_s * (2**attempt))
        result = await adapter.run(case)
    return result
