"""Build per-language hybrid indexes (dense HNSW/numpy + BM25)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

from app.config import ROOT, get_settings
from chunking.pipeline import build_chunks_for_passage
from chunking.types import BuiltChunk, RawPassage
from retrieval.dense import DenseIndex, faiss_available
from retrieval.embedder import Embedder
from retrieval.sparse import SparseIndex


def load_jsonl(path: Path) -> list[RawPassage]:
    rows: list[RawPassage] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                RawPassage(
                    query_id=row.get("query_id"),
                    query_type=row.get("query_type"),
                    language=row["language"],
                    passage_index=int(row.get("passage_index") or 0),
                    is_selected=int(row.get("is_selected") or 0),
                    source=row.get("source") or "translated",
                    text=row["text"],
                )
            )
    return rows


def load_demo() -> list[RawPassage]:
    return load_jsonl(ROOT / "data" / "demo" / "passages.jsonl")


class IndexBuilder:
    """Stream passages → chunks → embeddings → FAISS/BM25 without holding the corpus."""

    def __init__(
        self,
        language: str,
        out_dir: Path,
        embedder: Embedder,
        batch_size: int = 256,
        flush_every: int = 2048,
    ) -> None:
        self.language = language
        self.shard_dir = out_dir / language
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.batch_size = batch_size
        self.flush_every = flush_every
        self.dense: DenseIndex | None = None
        self.sparse = SparseIndex()
        self.buffer: list[BuiltChunk] = []
        self.n_passages = 0
        self.n_chunks = 0
        self.strategies: set[str] = set()
        self._jsonl = (self.shard_dir / "chunks.jsonl").open("wb")

    def add_passage(self, passage: RawPassage) -> None:
        self.n_passages += 1
        self.buffer.extend(build_chunks_for_passage(passage))
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        texts = [c.text for c in self.buffer]
        vectors = self.embedder.encode(
            texts, kind="passage", batch_size=self.batch_size, show_progress=False
        )
        if self.dense is None:
            self.dense = DenseIndex(dim=int(vectors.shape[1]), use_hnsw=True)
        self.dense.add(np.asarray(vectors, dtype=np.float32))
        self.sparse.add(texts)
        for chunk in self.buffer:
            self._jsonl.write(orjson.dumps(chunk.to_dict()) + b"\n")
            self.strategies.add(chunk.strategy)
        self.n_chunks += len(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self._jsonl.close()
        if self.dense is None or self.n_chunks == 0:
            print(f"skip {self.language}: no chunks")
            return
        self.dense.save(self.shard_dir)
        self.sparse.save(self.shard_dir)
        meta = {
            "language": self.language,
            "n_passages": self.n_passages,
            "n_chunks": self.n_chunks,
            "embedding_model": self.embedder.model_name,
            "dim": self.dense.dim,
            "dense_backend": "faiss_hnsw" if faiss_available() else "numpy",
            "strategies": sorted(self.strategies),
        }
        (self.shard_dir / "shard_meta.json").write_bytes(
            orjson.dumps(meta, option=orjson.OPT_INDENT_2)
        )
        print(f"  wrote {self.shard_dir} chunks={self.n_chunks} ({meta['dense_backend']})")


def write_shard(
    language: str,
    passages: Iterable[RawPassage],
    out_dir: Path,
    embedder: Embedder,
    batch_size: int,
) -> None:
    builder = IndexBuilder(language, out_dir, embedder, batch_size=batch_size)
    for passage in tqdm(passages, desc=f"index {language}", mininterval=2.0):
        builder.add_passage(passage)
    builder.close()


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build FAISS+BM25 shards")
    parser.add_argument("--demo", action="store_true", help="Index bundled demo corpus")
    parser.add_argument("--languages", default=",".join(settings.language_list))
    parser.add_argument("--split", default="train")
    parser.add_argument("--subsample-dir", type=Path, default=ROOT / "data" / "subsample")
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    embedder = Embedder()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from collections import defaultdict

        grouped: dict[str, list[RawPassage]] = defaultdict(list)
        for passage in load_demo():
            grouped[passage.language].append(passage)
        for lang, group in grouped.items():
            write_shard(lang, group, args.out, embedder, args.batch_size)
        return

    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    for lang in langs:
        path = args.subsample_dir / f"{lang}_{args.split}.jsonl"
        if not path.exists():
            print(f"missing {path} — run python -m ingest.full_train or ingest.subsample")
            continue
        write_shard(lang, load_jsonl(path), args.out, embedder, args.batch_size)


if __name__ == "__main__":
    main()
