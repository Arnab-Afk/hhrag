"""Build FAISS + BM25 from an existing chunks.jsonl without Hub downloads.

The full-train job wrote chunks but crashed before close(), so shard_meta
and vectors are still the tiny demo leftovers. This re-embeds locally from
disk (HF_HUB_OFFLINE) and overwrites dense/sparse artifacts only — it does
not rewrite chunks.jsonl.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

from app.config import get_settings
from retrieval.dense import DenseIndex, faiss_available
from retrieval.embedder import Embedder

# Fail fast instead of hanging on a Hub HEAD request during finalize.
# remaining.py unsets these before Hub parquet downloads.


def iter_chunk_rows(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield orjson.loads(line)


def estimate_lines(path: Path, sample: int = 2000) -> int:
    size = path.stat().st_size
    if size == 0:
        return 0
    n = 0
    nbytes = 0
    with path.open("rb") as handle:
        for line in handle:
            n += 1
            nbytes += len(line)
            if n >= sample:
                break
    if n == 0:
        return 0
    return max(n, int(size / (nbytes / n)))


def finalize_language(
    lang: str,
    index_dir: Path,
    embedder: Embedder,
    batch_size: int,
    flush_every: int,
    overwrite: bool,
) -> None:
    shard = index_dir / lang
    jsonl = shard / "chunks.jsonl"
    meta_path = shard / "shard_meta.json"
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")

    if meta_path.exists() and not overwrite:
        meta = orjson.loads(meta_path.read_bytes())
        n_chunks = int(meta.get("n_chunks") or 0)
        has_index = (shard / "faiss.index").exists() or (
            (shard / "vectors.npy").exists() and (shard / "vectors.npy").stat().st_size > 1_000_000
        )
        if n_chunks >= 10_000 and has_index:
            print(f"skip {lang}: already finalized n_chunks={n_chunks}")
            return

    est = estimate_lines(jsonl)
    print(f"\n=== finalize {lang} ===")
    print(f"  {jsonl}  {jsonl.stat().st_size / 1e9:.2f} GB  ~{est} chunks")
    print(f"  faiss={faiss_available()}  batch={batch_size}  flush={flush_every}")
    if not faiss_available():
        raise SystemExit(
            "FAISS is required for a full-train shard. "
            "pip install faiss-cpu in the venv, then retry."
        )

    dense = DenseIndex(dim=embedder.dim, use_hnsw=True)
    buffer: list[str] = []
    strategies: set[str] = set()
    n_chunks = 0
    parents: set[str] = set()

    def flush() -> None:
        nonlocal buffer, n_chunks
        if not buffer:
            return
        vectors = embedder.encode(
            buffer, kind="passage", batch_size=batch_size, show_progress=False
        )
        dense.add(np.asarray(vectors, dtype=np.float32))
        n_chunks += len(buffer)
        buffer = []

    for row in tqdm(iter_chunk_rows(jsonl), total=est, desc=lang, mininterval=2.0):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        buffer.append(text)
        strategies.add(row.get("strategy") or "passage_native")
        parent = row.get("parent_id")
        if parent:
            parents.add(parent)
        if len(buffer) >= flush_every:
            flush()
    flush()

    if n_chunks == 0:
        raise SystemExit(f"{lang}: no chunks in {jsonl}")

    print(f"  saving dense  chunks={n_chunks} passages~={len(parents)}")
    dense.save(shard)
    from retrieval.chunk_store import ensure_offsets

    ensure_offsets(shard)
    demo_json = shard / "chunks.json"
    if demo_json.exists() and demo_json.stat().st_size < 1_000_000:
        demo_json.unlink()
    npy = shard / "vectors.npy"
    if (shard / "faiss.index").exists() and npy.exists() and npy.stat().st_size < 1_000_000:
        npy.unlink()

    meta = {
        "language": lang,
        "n_passages": len(parents) or n_chunks,
        "n_chunks": n_chunks,
        "embedding_model": embedder.model_name,
        "dim": dense.dim,
        "dense_backend": "faiss_hnsw" if faiss_available() else "numpy",
        "strategies": sorted(strategies),
        "source": "finalize_jsonl",
    }
    meta_path.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    print(f"  wrote {shard} chunks={n_chunks} ({meta['dense_backend']})")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Finalize FAISS+BM25 from chunks.jsonl")
    parser.add_argument("--languages", default="hi,en,ta")
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--flush-every", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    embedder = Embedder()
    print(f"Embedding model: {embedder.model_name} dim={embedder.dim}")
    for lang in langs:
        finalize_language(
            lang,
            args.out,
            embedder,
            args.batch_size,
            args.flush_every,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
