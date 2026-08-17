"""Script-aware sentence splitting for Indic + English.

Naive `.` splitting mangles Hindi (danda `।`), abbreviations, and
decimal numbers. This splitter is the shared sentence layer used by
fixed-window, semantic, and parent-child chunkers.
"""

from __future__ import annotations

import re

from chunking.tokens import normalize_ws

DANDA = "\u0964"  # ।
DOUBLE_DANDA = "\u0965"  # ॥

# Devanagari / Bengali / Gurmukhi / Tamil / Telugu / etc. danda + latin.
_BOUNDARY = re.compile(
    rf"(?<=[{DANDA}{DOUBLE_DANDA}\.!\?؟۔])\s+"
)

# Protect common English abbreviations and decimals before splitting.
_ABBR = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|Fig|No|Rs|St)\.",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_ELLIPSIS = re.compile(r"\.{3,}")


def split_sentences(text: str, language: str | None = None) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []

    placeholders: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"¤{len(placeholders) - 1}¤"

    protected = _ELLIPSIS.sub(_hold, text)
    protected = _DECIMAL.sub(_hold, protected)
    protected = _ABBR.sub(_hold, protected)

    raw = _BOUNDARY.split(protected)
    sentences: list[str] = []
    for piece in raw:
        restored = piece
        for i, original in enumerate(placeholders):
            restored = restored.replace(f"¤{i}¤", original)
        restored = normalize_ws(restored)
        if restored:
            sentences.append(restored)

    # Very long "sentences" (tables, lists) still get a hard wrap later.
    if language in {"hi", "mr", "ne", "sa"} and len(sentences) == 1 and DANDA not in text:
        # Some translations omit danda; fall back to clause commas only if huge.
        if len(text) > 800:
            return _comma_wrap(text)
    return sentences or [text]


def _comma_wrap(text: str, target: int = 220) -> list[str]:
    parts = re.split(r"(?<=[,;،])\s+", text)
    buckets: list[str] = []
    current: list[str] = []
    size = 0
    for part in parts:
        current.append(part)
        size += len(part)
        if size >= target:
            buckets.append(" ".join(current))
            current, size = [], 0
    if current:
        buckets.append(" ".join(current))
    return buckets
