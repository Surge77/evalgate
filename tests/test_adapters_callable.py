import asyncio

import pytest

from evalgate.adapters.base import Adapter
from evalgate.adapters.callable import CallableAdapter
from evalgate.schemas import AgentResult, Case


def make_case() -> Case:
    return Case.model_validate(
        {"id": "case-001", "category": "qa", "input": {"q": "x"}}
    )


async def test_sync_callable_returning_str() -> None:
    adapter = CallableAdapter(lambda case: f"answer to {case.id}")
    result = await adapter.run(make_case())
    assert result.output == "answer to case-001"
    assert result.case_id == "case-001"
    assert result.error is None
    assert result.latency_ms >= 0


async def test_async_callable_returning_str() -> None:
    async def target(case: Case) -> str:
        await asyncio.sleep(0)
        return "async answer"

    result = await CallableAdapter(target).run(make_case())
    assert result.output == "async answer"


async def test_callable_returning_agent_result_keeps_metadata() -> None:
    def target(case: Case) -> AgentResult:
        return AgentResult(
            case_id="wrong-id", output="rich", cost_usd=0.01, latency_ms=5.0
        )

    result = await CallableAdapter(target).run(make_case())
    assert result.case_id == "case-001"
    assert result.cost_usd == 0.01
    assert result.latency_ms == 5.0


async def test_exception_becomes_error_result() -> None:
    def target(case: Case) -> str:
        raise RuntimeError("target exploded")

    result = await CallableAdapter(target).run(make_case())
    assert result.error == "target exploded"
    assert result.output == ""


def test_satisfies_adapter_protocol() -> None:
    adapter = CallableAdapter(lambda case: "x")
    assert isinstance(adapter, Adapter)


def test_from_spec_loads_module_attribute() -> None:
    adapter = CallableAdapter.from_spec("tests.targets:echo_question")
    assert adapter.name == "tests.targets:echo_question"


@pytest.mark.parametrize("spec", ["no-colon", ":attr", "module:"])
def test_from_spec_rejects_malformed(spec: str) -> None:
    with pytest.raises(ValueError, match="invalid callable spec"):
        CallableAdapter.from_spec(spec)
