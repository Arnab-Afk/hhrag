"""Input-side guardrails: empty, garbage, unsafe, obviously off-brief."""

from __future__ import annotations

import re
import unicodedata

from app.schemas import GuardrailEvent

_UNSAFE = re.compile(
    r"("
    r"bomb making|build a bomb|how to make a bomb|"
    r"credit card (dump|cvv)|steal (ssn|aadhaar|password)|"
    r"child (porn|sexual)|csam|"
    r"kill (myself|himself|herself)|suicide (method|how to)|"
    r"बम बनाना|आत्महत्या कैसे"
    r")",
    re.IGNORECASE,
)

_GARBAGE = re.compile(r"^(.)\1{8,}$")
_ONLY_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def check_input(query: str) -> GuardrailEvent:
    text = unicodedata.normalize("NFKC", query or "").strip()
    if len(text) < 3:
        return GuardrailEvent(
            name="input",
            passed=False,
            reason="empty_or_too_short",
            details={"length": len(text)},
        )
    if _ONLY_PUNCT.match(text) or _GARBAGE.match(text):
        return GuardrailEvent(
            name="input",
            passed=False,
            reason="garbage_transcript",
        )
    if _UNSAFE.search(text):
        return GuardrailEvent(
            name="input",
            passed=False,
            reason="unsafe_query",
        )
    return GuardrailEvent(name="input", passed=True, reason="ok")
