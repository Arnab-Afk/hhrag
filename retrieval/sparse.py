from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunking.tokens import lexical_tokens


class SparseIndex:
    def __init__(self, corpus_tokens: list[list[str]] | None = None) -> None:
        self._tokens = corpus_tokens or []
        self._bm25: BM25Okapi | None = None
        if self._tokens:
            self._bm25 = BM25Okapi(self._tokens)

    def add(self, documents: list[str]) -> None:
        self._tokens.extend(lexical_tokens(doc) for doc in documents)
        self._bm25 = BM25Okapi(self._tokens)

    @property
    def ntotal(self) -> int:
        return len(self._tokens)

    def search(self, query: str, top_k: int) -> tuple[list[float], list[int]]:
        if not self._bm25 or not self._tokens:
            return [], []
        q = lexical_tokens(query)
        if not q:
            return [], []
        scores = self._bm25.get_scores(q)
        k = min(top_k, len(scores))
        import numpy as np

        order = np.argsort(-np.asarray(scores))[:k]
        return [float(scores[i]) for i in order], [int(i) for i in order]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "bm25.pkl").open("wb") as handle:
            pickle.dump(self._tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        with (path / "bm25.pkl").open("rb") as handle:
            tokens = pickle.load(handle)
        return cls(corpus_tokens=tokens)
