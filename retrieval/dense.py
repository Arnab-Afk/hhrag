"""In-memory dense index: FAISS HNSW when available, numpy otherwise.

Windows pip often lacks faiss-cpu wheels; brute-force cosine on 20–80k
× 384-d is still well under the 200ms retrieval budget on a laptop CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import orjson

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional
    faiss = None
    _HAS_FAISS = False


class DenseIndex:
    def __init__(self, dim: int, use_hnsw: bool = True) -> None:
        self.dim = dim
        self.use_hnsw = use_hnsw and _HAS_FAISS
        self._vectors: np.ndarray | None = None
        self._index = None
        if self.use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            self._index = index

    @property
    def ntotal(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._index is not None:
            self._index.add(vectors)
            return
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        k = min(top_k, max(1, self.ntotal))
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return scores[0], ids[0]
        if self._vectors is None or self._vectors.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        scores = self._vectors @ q.reshape(-1)
        if k >= scores.shape[0]:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return scores[order].astype(np.float32), order.astype(np.int64)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "backend": "faiss_hnsw" if self._index is not None else "numpy",
            "ntotal": self.ntotal,
        }
        (path / "dense_meta.json").write_bytes(orjson.dumps(meta))
        if self._index is not None:
            faiss.write_index(self._index, str(path / "faiss.index"))
            return
        assert self._vectors is not None
        np.save(path / "vectors.npy", self._vectors)

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        meta = orjson.loads((path / "dense_meta.json").read_bytes())
        dim = int(meta["dim"])
        backend = meta.get("backend", "numpy")
        store = cls(dim=dim, use_hnsw=backend.startswith("faiss") and _HAS_FAISS)
        faiss_path = path / "faiss.index"
        npy_path = path / "vectors.npy"
        if backend.startswith("faiss") and _HAS_FAISS and faiss_path.exists():
            store._index = faiss.read_index(str(faiss_path))
            store.use_hnsw = True
            return store
        if npy_path.exists():
            store.use_hnsw = False
            store._index = None
            store._vectors = np.load(npy_path).astype(np.float32)
            return store
        if faiss_path.exists() and _HAS_FAISS:
            store._index = faiss.read_index(str(faiss_path))
            return store
        raise FileNotFoundError(f"No dense index found in {path}")


def faiss_available() -> bool:
    return _HAS_FAISS
