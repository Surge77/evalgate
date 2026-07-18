"""Adapter protocol — anything that can answer a Case."""

from typing import Protocol, runtime_checkable

from evalgate.schemas import AgentResult, Case


@runtime_checkable
class Adapter(Protocol):
    name: str

    async def run(self, case: Case) -> AgentResult: ...
