"""Reciprocal rank fusion of dense + sparse, with metadata-aware boosts."""

from __future__ import annotations

from collections import defaultdict

from app.schemas import Chunk
from retrieval.query_type import guess_query_type


def rrf_fuse(
    dense_ids: list[int],
    sparse_ids: list[int],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_ids, start=1):
        scores[idx] += dense_weight * (1.0 / (k + rank))
    for rank, idx in enumerate(sparse_ids, start=1):
        scores[idx] += sparse_weight * (1.0 / (k + rank))
    return dict(scores)


def apply_metadata_boost(
    chunks: list[Chunk],
    query: str,
) -> list[Chunk]:
    """Boost NUMERIC/PERSON chunks when the query looks like that type.

    is_selected is stored but NEVER used as a retrieval signal — that
    would leak MSMARCO labels into the live system.
    """
    guess = guess_query_type(query)
    for chunk in chunks:
        boost = 0.0
        if chunk.query_type and chunk.query_type.upper() == guess:
            boost += 0.08
        if guess == "NUMERIC" and any(ch.isdigit() for ch in chunk.text):
            boost += 0.04
        chunk.score = chunk.score + boost
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks
