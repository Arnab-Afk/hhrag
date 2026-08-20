"""Random-access into chunks.jsonl via a byte-offset sidecar.

Full-train shards are 6–18GB. Parsing them into a Python list blows RAM
and blocks the API for minutes. Offsets are ~8 bytes × n_chunks (~56MB)
and let search seek only the FAISS hit rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import orjson

OFFSETS_NAME = "chunks.offsets.npy"


def offsets_path(shard_dir: Path) -> Path:
    return shard_dir / OFFSETS_NAME


def build_line_offsets(jsonl: Path) -> np.ndarray:
    """Byte offset of every row that was actually embedded (non-empty text).

    Must match ingest.finalize_jsonl / IndexBuilder, which skip blank text.
    """
    offsets: list[int] = []
    pos = 0
    with jsonl.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = orjson.loads(stripped)
                if (row.get("text") or "").strip():
                    offsets.append(pos)
            pos += len(line)
    return np.asarray(offsets, dtype=np.uint64)


def ensure_offsets(shard_dir: Path) -> np.ndarray:
    jsonl = shard_dir / "chunks.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"No chunks.jsonl in {shard_dir}")
    out = offsets_path(shard_dir)
    if out.exists() and out.stat().st_mtime >= jsonl.stat().st_mtime:
        return np.load(out, mmap_mode="r")
    arr = build_line_offsets(jsonl)
    np.save(out, arr)
    return arr


def read_rows(jsonl: Path, offsets: np.ndarray, ids: list[int]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    with jsonl.open("rb") as handle:
        for idx in ids:
            if idx < 0 or idx >= len(offsets):
                continue
            handle.seek(int(offsets[idx]))
            line = handle.readline()
            if not line.strip():
                continue
            found[idx] = orjson.loads(line)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunks.offsets.npy sidecars")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--languages", default="")
    args = parser.parse_args()
    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    children = (
        [args.index_dir / lang for lang in langs]
        if langs
        else sorted(p for p in args.index_dir.iterdir() if p.is_dir())
    )
    for shard in children:
        jsonl = shard / "chunks.jsonl"
        if not jsonl.exists():
            continue
        print(f"offsets {shard.name} ({jsonl.stat().st_size / 1e9:.2f} GB)...")
        arr = ensure_offsets(shard)
        print(f"  {len(arr)} lines -> {offsets_path(shard)}")


if __name__ == "__main__":
    main()
