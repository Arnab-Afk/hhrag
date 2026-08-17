"""Post-generation grounding: citations must exist; answer must overlap context.

This is a cheap lexical+character overlap check, not a full NLI model.
It is meant to catch obvious hallucination, then refuse or downgrade
rather than ship an ungrounded answer.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import Chunk, GenResult, GuardrailEvent
from chunking.semantic import cosine_ngram
from chunking.tokens import lexical_tokens


def _overlap(answer: str, context: str) -> float:
    a = set(lexical_tokens(answer))
    b = set(lexical_tokens(context))
    if not a or not b:
        return cosine_ngram(answer, context)
    jaccard = len(a & b) / len(a)
    char = cosine_ngram(answer, context)
    return 0.65 * jaccard + 0.35 * char


def check_grounding(
    generation: GenResult,
    chunks: list[Chunk],
    min_overlap: float | None = None,
) -> GuardrailEvent:
    settings = get_settings()
    threshold = min_overlap if min_overlap is not None else settings.grounding_min_overlap
    allowed = {c.chunk_id for c in chunks} | {c.parent_id for c in chunks}
    cited = generation.citation_ids
    unknown = [cid for cid in cited if cid not in allowed]
    if unknown:
        return GuardrailEvent(
            name="grounding",
            passed=False,
            reason="unknown_citations",
            details={"unknown": unknown},
        )

    cited_chunks = [c for c in chunks if c.chunk_id in cited or c.parent_id in cited]
    context_pool = cited_chunks or chunks
    context = "\n".join(c.parent_text for c in context_pool[:6])
    overlap = _overlap(generation.answer, context)
    passed = overlap >= threshold and bool(generation.answer.strip())
    return GuardrailEvent(
        name="grounding",
        passed=passed,
        reason="ok" if passed else "low_overlap",
        details={"overlap": round(overlap, 4), "threshold": threshold},
    )
