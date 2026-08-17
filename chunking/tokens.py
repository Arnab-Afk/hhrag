from __future__ import annotations

import re

# Approximate tokens for Indic + English. Whitespace is meaningful in all
# MSMARCO-XI target languages we index (hi, en, ta, bn, mr).
_PUNCT = re.compile(r"[^\w\u0900-\u0D7F\u0A00-\u0A7F\u0980-\u09FF]+", re.UNICODE)


def approx_tokens(text: str) -> int:
    words = [w for w in text.split() if w]
    if not words:
        return max(1, len(text) // 6) if text else 0
    return max(len(words), len(text) // 8)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def lexical_tokens(text: str) -> list[str]:
    lowered = text.lower()
    parts = _PUNCT.split(lowered)
    return [p for p in parts if len(p) > 1]
