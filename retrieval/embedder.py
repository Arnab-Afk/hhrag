"""Local multilingual embeddings.

e5 models expect `query: ` / `passage: ` prefixes. The first call
downloads the model (~400MB for e5-small) into the HuggingFace cache.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from app.config import get_settings


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dim: int | None = None

    @property
    def uses_e5_prefix(self) -> bool:
        return "e5" in self.model_name.lower()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def _prepare(self, texts: Sequence[str], kind: str) -> list[str]:
        if not self.uses_e5_prefix:
            return list(texts)
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: str = "passage",
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        model = self._load()
        prepared = self._prepare(texts, kind)
        vectors = model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], kind="query", batch_size=1)[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
