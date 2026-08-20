from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from app.config import Settings, get_settings
from app.schemas import Chunk, RetrievalResult
from retrieval.chunk_store import ensure_offsets, read_rows
from retrieval.dense import DenseIndex
from retrieval.embedder import Embedder, get_embedder
from retrieval.fusion import apply_metadata_boost, rrf_fuse
from retrieval.language import shard_for_language_code
from retrieval.query_type import guess_query_type
from retrieval.sparse import rescore_bm25

# Dense shortlist size before BM25 rerank. Must stay tiny vs corpus size.
SPARSE_CANDIDATES = 50

logger = logging.getLogger(__name__)


def _row_to_chunk(row: dict[str, Any], score: float) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        parent_id=row["parent_id"],
        text=row["text"],
        parent_text=row.get("parent_text") or row["text"],
        language=row["language"],
        query_id=row.get("query_id"),
        query_type=row.get("query_type"),
        is_selected=int(row.get("is_selected") or 0),
        source=row.get("source") or "translated",
        strategy=row.get("strategy") or "passage_native",
        score=score,
    )


class LanguageShard:
    def __init__(
        self,
        language: str,
        dense: DenseIndex,
        *,
        rows: list[dict[str, Any]] | None = None,
        jsonl: Path | None = None,
        offsets: np.ndarray | None = None,
    ) -> None:
        self.language = language
        self.dense = dense
        self._rows = rows
        self._jsonl = jsonl
        self._offsets = offsets

    def _fetch(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        found: dict[int, dict[str, Any]] = {}
        if self._rows is not None:
            for idx in ids:
                if 0 <= idx < len(self._rows):
                    found[idx] = self._rows[idx]
            return found
        if self._jsonl is None or self._offsets is None:
            return found
        return read_rows(self._jsonl, self._offsets, ids)

    @classmethod
    def load(cls, path: Path) -> "LanguageShard":
        meta = orjson.loads((path / "shard_meta.json").read_bytes())
        jsonl = path / "chunks.jsonl"
        blob = path / "chunks.json"
        dense = DenseIndex.load(path)
        if jsonl.exists():
            offsets = ensure_offsets(path)
            logger.info(
                "Shard %s: faiss ntotal=%s jsonl offsets=%s",
                meta["language"],
                dense.ntotal,
                len(offsets),
            )
            if dense.ntotal and abs(int(dense.ntotal) - len(offsets)) > 1:
                logger.warning(
                    "Shard %s offset/faiss mismatch: offsets=%s ntotal=%s",
                    meta["language"],
                    len(offsets),
                    dense.ntotal,
                )
            return cls(
                language=meta["language"],
                dense=dense,
                jsonl=jsonl,
                offsets=offsets,
            )
        if blob.exists():
            rows = orjson.loads(blob.read_bytes())
            return cls(language=meta["language"], dense=dense, rows=rows)
        raise FileNotFoundError(f"No chunks.jsonl or chunks.json in {path}")

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[Chunk]:
        candidate_k = max(top_k, SPARSE_CANDIDATES)
        _d_scores, d_ids = self.dense.search(query_vec, candidate_k)
        dense_ids = [int(i) for i in d_ids if i >= 0]
        if not dense_ids:
            return []
        by_id = self._fetch(dense_ids)
        ordered_ids = [i for i in dense_ids if i in by_id]
        texts = [by_id[i]["text"] for i in ordered_ids]
        local_rank = rescore_bm25(query, texts)
        sparse_ids = [ordered_ids[i] for i in local_rank]
        fused = rrf_fuse(
            ordered_ids,
            sparse_ids,
            k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        dense_rank = {int(i): r + 1 for r, i in enumerate(ordered_ids)}
        sparse_rank = {int(i): r + 1 for r, i in enumerate(sparse_ids)}
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[Chunk] = []
        for idx, score in ranked:
            row = by_id.get(idx)
            if row is None:
                continue
            chunk = _row_to_chunk(row, score)
            chunk.dense_rank = dense_rank.get(idx)
            chunk.sparse_rank = sparse_rank.get(idx)
            out.append(chunk)
        return out


class HybridRetriever:
    def __init__(
        self,
        shard_dirs: dict[str, Path],
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.shard_dirs = shard_dirs
        self._cache: dict[str, LanguageShard] = {}
        self.embedder = embedder or get_embedder()
        self.settings = settings or get_settings()

    @property
    def shards(self) -> dict[str, Path]:
        return self.shard_dirs

    def _shard(self, name: str) -> LanguageShard:
        if name not in self._cache:
            self._cache[name] = LanguageShard.load(self.shard_dirs[name])
            logger.info("Loaded shard %s", name)
        return self._cache[name]

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "HybridRetriever":
        settings = get_settings()
        root = Path(index_dir or settings.index_dir)
        shard_dirs: dict[str, Path] = {}
        if not root.exists():
            raise FileNotFoundError(
                f"No indexes at {root}. Run: python -m ingest.build_index --demo"
            )
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "shard_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = orjson.loads(meta_path.read_bytes())
            except Exception:
                continue
            n_chunks = int(meta.get("n_chunks") or 0)
            jsonl = child / "chunks.jsonl"
            # Stale demo meta sitting on a full-train jsonl — skip until finalized.
            if jsonl.exists() and jsonl.stat().st_size > 50_000_000 and n_chunks < 1_000:
                continue
            if not (child / "faiss.index").exists() and not (child / "vectors.npy").exists():
                continue
            if (child / "chunks.jsonl").exists() or (child / "chunks.json").exists():
                shard_dirs[child.name] = child
        if not shard_dirs:
            raise FileNotFoundError(f"No language shards in {root}")
        logger.info("Shard dirs: %s", ", ".join(shard_dirs))
        return cls(shard_dirs=shard_dirs, settings=settings)

    def _plan_shards(self, language_code: str | None) -> list[str]:
        preferred = shard_for_language_code(language_code)
        if preferred and preferred in self.shard_dirs:
            plan = [preferred]
            if preferred != "en" and "en" in self.shard_dirs:
                plan.append("en")
            return plan
        plan = [name for name in ("hi", "en") if name in self.shard_dirs]
        for name in self.shard_dirs:
            if name not in plan:
                plan.append(name)
        return plan

    def retrieve(
        self,
        query: str,
        language_code: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        settings = self.settings
        top_k = top_k or settings.retrieve_top_k
        plan = self._plan_shards(language_code)
        query_vec = self.embedder.encode_query(query)

        collected: list[Chunk] = []
        searched: list[str] = []
        for i, shard_name in enumerate(plan):
            shard = self._shard(shard_name)
            searched.append(shard_name)
            hits = shard.search(
                query,
                query_vec,
                top_k=top_k,
                rrf_k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )
            collected.extend(hits)
            best = max((c.score for c in collected), default=0.0)
            if i == 0 and best >= settings.min_retrieval_score and len(plan) > 1:
                break

        collected = apply_metadata_boost(collected, query)
        unique: list[Chunk] = []
        seen_parents: set[str] = set()
        for chunk in collected:
            if chunk.parent_id in seen_parents:
                continue
            seen_parents.add(chunk.parent_id)
            unique.append(chunk)
            if len(unique) >= min(8, top_k):
                break

        max_score = unique[0].score if unique else 0.0
        preferred = shard_for_language_code(language_code)
        lang = preferred or (unique[0].language if unique else (plan[0] if plan else "en"))
        return RetrievalResult(
            chunks=unique,
            max_score=max_score,
            language=lang,
            shards_searched=searched,
            query_type_guess=guess_query_type(query),
        )
