"""Compose passage-native, overlapping windows, semantic splits, parent-child.

MSMARCO-XI passages are already retrieval units. We still run a real
chunking stack because:

1. Selected gold passages are short; keep them whole (passage-native).
2. Long unselected / concatenated translations need windows + overlap.
3. Semantic splits catch topic shifts the window would glue together.
4. Parent-child embeds the small unit but returns the full passage to
   the generator — better matching without starving grounding context.
5. Metadata (language, query_type, is_selected, source) rides on every
   chunk for filter/boost at retrieval time.
6. Simhash dedup collapses the near-duplicate passage soup MSMARCO is
   famous for, shrinking ANN search space.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.config import Settings, get_settings
from chunking.dedup import dedup_texts
from chunking.fixed import window_chunks
from chunking.semantic import semantic_chunks
from chunking.tokens import approx_tokens, normalize_ws
from chunking.types import BuiltChunk, RawPassage


def build_chunks_for_passage(
    passage: RawPassage,
    settings: Settings | None = None,
) -> list[BuiltChunk]:
    settings = settings or get_settings()
    parent_text = normalize_ws(passage.text)
    if not parent_text:
        return []

    parent_id = (
        f"{passage.language}_q{passage.query_id}_p{passage.passage_index}_{passage.source}"
    )
    token_count = approx_tokens(parent_text)

    if token_count <= settings.long_passage_tokens:
        child_texts = [parent_text]
        strategy = "passage_native"
    else:
        windows = window_chunks(
            parent_text,
            passage.language,
            window_tokens=settings.window_tokens,
            overlap_tokens=settings.overlap_tokens,
        )
        semantic = semantic_chunks(
            parent_text,
            passage.language,
            threshold=settings.semantic_sim_threshold,
            max_tokens=settings.window_tokens,
        )
        # Union, prefer semantic units; fall back to windows if semantic
        # collapsed to a single over-long blob.
        child_texts = _unique_keep_order(semantic + windows)
        strategy = "semantic+overlap"

    chunks: list[BuiltChunk] = []
    for i, child in enumerate(child_texts):
        chunks.append(
            BuiltChunk(
                chunk_id=f"{parent_id}_c{i}",
                parent_id=parent_id,
                text=child,
                parent_text=parent_text,
                language=passage.language,
                query_id=passage.query_id,
                query_type=passage.query_type,
                is_selected=passage.is_selected,
                source=passage.source,
                strategy=strategy if i == 0 or token_count > settings.long_passage_tokens else "passage_native",
            )
        )
    return chunks


def chunk_corpus(
    passages: Sequence[RawPassage] | Iterable[RawPassage],
    settings: Settings | None = None,
) -> list[BuiltChunk]:
    settings = settings or get_settings()
    passages = list(passages)
    keep = dedup_texts((p.text for p in passages), max_hamming=settings.simhash_hamming)
    kept = [passages[i] for i in keep]

    out: list[BuiltChunk] = []
    for passage in kept:
        out.extend(build_chunks_for_passage(passage, settings))
    return out


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique
