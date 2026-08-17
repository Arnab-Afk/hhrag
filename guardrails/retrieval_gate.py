from __future__ import annotations

from app.config import get_settings
from app.schemas import GuardrailEvent, RetrievalResult


def check_retrieval(
    retrieval: RetrievalResult,
    min_score: float | None = None,
) -> GuardrailEvent:
    threshold = min_score if min_score is not None else get_settings().min_retrieval_score
    if not retrieval.chunks:
        return GuardrailEvent(
            name="retrieval_gate",
            passed=False,
            reason="no_hits",
            details={"max_score": 0.0, "threshold": threshold},
        )
    if retrieval.max_score < threshold:
        return GuardrailEvent(
            name="retrieval_gate",
            passed=False,
            reason="below_min_score",
            details={
                "max_score": round(retrieval.max_score, 5),
                "threshold": threshold,
            },
        )
    return GuardrailEvent(
        name="retrieval_gate",
        passed=True,
        reason="ok",
        details={"max_score": round(retrieval.max_score, 5), "threshold": threshold},
    )
