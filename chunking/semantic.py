"""Adaptive / semantic splitting on consecutive-sentence similarity.

Uses cheap character-ngram cosine so ingest does not need a second
embedding model. Topic-shift = similarity drop below threshold.
"""

from __future__ import annotations

import math
from collections import Counter

from chunking.script_aware import split_sentences
from chunking.tokens import approx_tokens


def _char_ngrams(text: str, n: int = 3) -> Counter[str]:
    padded = f" {text.lower()} "
    return Counter(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))


def cosine_ngram(a: str, b: str) -> float:
    ca, cb = _char_ngrams(a), _char_ngrams(b)
    if not ca or not cb:
        return 0.0
    dot = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_chunks(
    text: str,
    language: str,
    threshold: float = 0.42,
    max_tokens: int = 128,
) -> list[str]:
    sentences = split_sentences(text, language)
    if len(sentences) <= 1:
        return sentences or []

    chunks: list[str] = []
    current = [sentences[0]]
    current_tokens = approx_tokens(sentences[0])
    for prev, sent in zip(sentences, sentences[1:]):
        sim = cosine_ngram(prev, sent)
        sent_tokens = approx_tokens(sent)
        topic_shift = sim < threshold
        too_long = current_tokens + sent_tokens > max_tokens
        if topic_shift or too_long:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks
