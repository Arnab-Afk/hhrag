"""Build per-language hybrid indexes (dense HNSW/numpy + BM25)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

from app.config import ROOT, get_settings
from chunking.pipeline import chunk_corpus
from chunking.types import RawPassage
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
    demo = ROOT / "data" / "demo" / "passages.jsonl"
    return load_jsonl(demo)


def write_shard(
    language: str,
    passages: list[RawPassage],
    out_dir: Path,
    embedder: Embedder,
    batch_size: int,
) -> None:
    chunks = chunk_corpus(passages)
    if not chunks:
        print(f"skip {language}: no chunks")
        return
    shard_dir = out_dir / language
    shard_dir.mkdir(parents=True, exist_ok=True)

    texts = [c.text for c in chunks]
    print(f"{language}: {len(passages)} passages -> {len(chunks)} chunks, embedding...")
    vectors = embedder.encode(texts, kind="passage", batch_size=batch_size, show_progress=True)

    dense = DenseIndex(dim=vectors.shape[1], use_hnsw=True)
    dense.add(np.asarray(vectors, dtype=np.float32))
    dense.save(shard_dir)

    sparse = SparseIndex()
    sparse.add(texts)
    sparse.save(shard_dir)

    payload = [c.to_dict() for c in chunks]
    (shard_dir / "chunks.json").write_bytes(orjson.dumps(payload))
    meta = {
        "language": language,
        "n_passages": len(passages),
        "n_chunks": len(chunks),
        "embedding_model": embedder.model_name,
        "dim": int(vectors.shape[1]),
        "dense_backend": "faiss_hnsw" if faiss_available() else "numpy",
        "strategies": sorted({c.strategy for c in chunks}),
    }
    (shard_dir / "shard_meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    print(f"  wrote {shard_dir} ({meta['dense_backend']})")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build FAISS+BM25 shards")
    parser.add_argument("--demo", action="store_true", help="Index bundled demo corpus")
    parser.add_argument("--languages", default=",".join(settings.language_list))
    parser.add_argument("--split", default="train")
    parser.add_argument("--subsample-dir", type=Path, default=ROOT / "data" / "subsample")
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    embedder = Embedder()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        passages = load_demo()
        by_lang: dict[str, list[RawPassage]] = defaultdict(list)
        for p in passages:
            by_lang[p.language].append(p)
        for lang, group in by_lang.items():
            write_shard(lang, group, args.out, embedder, args.batch_size)
        return

    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    for lang in langs:
        path = args.subsample_dir / f"{lang}_{args.split}.jsonl"
        if not path.exists():
            print(f"missing {path} — run python -m ingest.subsample first")
            continue
        write_shard(lang, load_jsonl(path), args.out, embedder, args.batch_size)


if __name__ == "__main__":
    main()
