"""Stub eval targets used by tests and offline demo runs."""

from evalgate.schemas import Case


def echo_question(case: Case) -> str:
    return str(case.input.get("question", ""))


def golden_parrot(case: Case) -> str:
    """Returns the reference verbatim — a perfect-score target for wiring tests."""
    return case.reference or ""
