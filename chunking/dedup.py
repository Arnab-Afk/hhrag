"""64-bit simhash over character 5-grams. Near-duplicates share prefixes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", _NON_ALNUM.sub(" ", text.lower())).strip()


def simhash64(text: str) -> int:
    vector = [0] * 64
    blob = f" {_fold(text)} "
    if len(blob) < 7:
        digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")
    for i in range(len(blob) - 4):
        gram = blob[i : i + 5].encode("utf-8")
        h = int.from_bytes(hashlib.blake2b(gram, digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, val in enumerate(vector):
        if val >= 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedup_texts(
    texts: Iterable[str],
    max_hamming: int = 3,
) -> list[int]:
    """Return indices of texts to keep (first occurrence wins)."""
    kept_hashes: list[int] = []
    keep: list[int] = []
    for idx, text in enumerate(texts):
        h = simhash64(text or "")
        if any(hamming(h, prev) <= max_hamming for prev in kept_hashes):
            continue
        kept_hashes.append(h)
        keep.append(idx)
    return keep
