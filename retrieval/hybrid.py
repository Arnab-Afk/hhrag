from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson

from app.config import Settings, get_settings
from app.schemas import Chunk, RetrievalResult
from retrieval.dense import DenseIndex
from retrieval.embedder import Embedder, get_embedder
from retrieval.fusion import apply_metadata_boost, rrf_fuse
from retrieval.language import shard_for_language_code
from retrieval.query_type import guess_query_type
from retrieval.sparse import SparseIndex

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
        chunks: list[dict[str, Any]],
        dense: DenseIndex,
        sparse: SparseIndex,
    ) -> None:
        self.language = language
        self.chunks = chunks
        self.dense = dense
        self.sparse = sparse

    @classmethod
    def load(cls, path: Path) -> "LanguageShard":
        meta = orjson.loads((path / "shard_meta.json").read_bytes())
        rows = orjson.loads((path / "chunks.json").read_bytes())
        dense = DenseIndex.load(path)
        sparse = SparseIndex.load(path)
        return cls(language=meta["language"], chunks=rows, dense=dense, sparse=sparse)

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[Chunk]:
        d_scores, d_ids = self.dense.search(query_vec, top_k)
        s_scores, s_ids = self.sparse.search(query, top_k)
        dense_ids = [int(i) for i in d_ids if i >= 0]
        fused = rrf_fuse(
            dense_ids,
            list(s_ids),
            k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        dense_rank = {int(i): r + 1 for r, i in enumerate(dense_ids)}
        sparse_rank = {int(i): r + 1 for r, i in enumerate(s_ids)}
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[Chunk] = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = _row_to_chunk(self.chunks[idx], score)
            chunk.dense_rank = dense_rank.get(idx)
            chunk.sparse_rank = sparse_rank.get(idx)
            out.append(chunk)
        return out


class HybridRetriever:
    def __init__(
        self,
        shards: dict[str, LanguageShard],
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.shards = shards
        self.embedder = embedder or get_embedder()
        self.settings = settings or get_settings()

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "HybridRetriever":
        settings = get_settings()
        root = Path(index_dir or settings.index_dir)
        shards: dict[str, LanguageShard] = {}
        if not root.exists():
            raise FileNotFoundError(
                f"No indexes at {root}. Run: python -m ingest.build_index --demo"
            )
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "chunks.json").exists():
                shards[child.name] = LanguageShard.load(child)
        if not shards:
            raise FileNotFoundError(f"No language shards in {root}")
        logger.info("Loaded shards: %s", ", ".join(shards))
        return cls(shards=shards, settings=settings)

    def _plan_shards(self, language_code: str | None) -> list[str]:
        preferred = shard_for_language_code(language_code)
        if preferred and preferred in self.shards:
            plan = [preferred]
            if preferred != "en" and "en" in self.shards:
                plan.append("en")
            return plan
        # Unknown language: prefer Hindi+English if present, else all.
        plan = [name for name in ("hi", "en") if name in self.shards]
        for name in self.shards:
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
            shard = self.shards[shard_name]
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
            # Skip remaining shards (e.g. English fallback) if the
            # language-routed shard already cleared the gate.
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
