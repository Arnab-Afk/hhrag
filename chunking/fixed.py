"""Fixed-size windows with overlap, packed on sentence boundaries."""

from __future__ import annotations

from chunking.script_aware import split_sentences
from chunking.tokens import approx_tokens


def window_chunks(
    text: str,
    language: str,
    window_tokens: int = 128,
    overlap_tokens: int = 24,
) -> list[str]:
    sentences = split_sentences(text, language)
    if not sentences:
        return []

    sizes = [approx_tokens(s) for s in sentences]
    if sum(sizes) <= window_tokens:
        return [" ".join(sentences)]

    chunks: list[str] = []
    start = 0
    n = len(sentences)
    while start < n:
        token_count = 0
        end = start
        while end < n and token_count + sizes[end] <= window_tokens:
            token_count += sizes[end]
            end += 1
        if end == start:
            end = start + 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= n:
            break
        # Walk back until we have ~overlap tokens of context.
        overlap = 0
        back = end - 1
        while back > start and overlap < overlap_tokens:
            overlap += sizes[back]
            back -= 1
        start = max(back, start + 1)
    return chunks
