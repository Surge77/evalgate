"""Deterministic demo target with seeded flaws.

Answers the golden_v1 dataset mostly correctly, but a fixed set of case ids
gets flawed answers. The flaw map is the construction ground truth behind
datasets/calibration/labels_v1.jsonl — keep the two in sync.
"""

from evalgate.schemas import AgentResult, Case

# Wrong answers: rule checks fail, correctness should score low.
WRONG_ANSWERS: dict[str, str] = {
    "qa-002": "Password reset links are valid for 12 hours.",
    "qa-009": "The Enterprise SLA guarantees 99.5% uptime.",
    "ex-003": '{"product": "USB-C Cable", "quantity": "three"}',
    "ma-004": "3.5 times 4 equals 12.",
    "cl-002": "neutral",
}

# Ungrounded answers: rule checks still pass, but the answer invents facts
# absent from the context — the flaw only the judge's faithfulness catches.
UNGROUNDED_ANSWERS: dict[str, str] = {
    "qa-001": (
        "Pro plan purchases can be refunded within 14 days of purchase, "
        "and enterprise customers get a full year."
    ),
    "qa-006": (
        "The API allows 120 requests per minute per key, "
        "which doubles automatically on weekends."
    ),
    "qa-010": "Data can be exported as CSV or JSON, as well as XML.",
    "qa-012": (
        "Billing cycle changes take effect at the next renewal date "
        "and incur a $5 processing fee."
    ),
}

# Off-topic answers: everything fails, relevance should score lowest.
OFFTOPIC_ANSWERS: dict[str, str] = {
    "qa-005": "Our Pro plan is very competitively priced against rivals.",
    "cl-005": "The weather is lovely today.",
}

# Wrong tool plans: tools called in the wrong order.
WRONG_TOOL_PLANS: dict[str, list[str]] = {
    "tp-003": ["summarize", "web_fetch"],
}


def answer(case: Case) -> AgentResult | str:
    for flaw_map in (WRONG_ANSWERS, UNGROUNDED_ANSWERS, OFFTOPIC_ANSWERS):
        if case.id in flaw_map:
            return flaw_map[case.id]
    if case.category == "tool_plan":
        tools = WRONG_TOOL_PLANS.get(
            case.id,
            [name.strip() for name in (case.reference or "").split(",")],
        )
        return AgentResult(
            case_id=case.id, output=", ".join(tools), tool_calls=tools
        )
    return case.reference or ""
