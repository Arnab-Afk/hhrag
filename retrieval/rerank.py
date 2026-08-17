"""Optional cross-encoder rerank. Off by default — it usually blows the 200ms budget."""

from __future__ import annotations

from app.schemas import Chunk


def rerank(query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return chunks[:top_n]
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(query, c.parent_text) for c in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:top_n]
