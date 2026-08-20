import json
from pathlib import Path

root = Path(r"D:\projects\hhrag\indexes")
print("root", root, "exists", root.exists())
if not root.exists():
    raise SystemExit(0)
for shard in sorted(p for p in root.iterdir() if p.is_dir()):
    print(f"\n== {shard.name} ==")
    for name in [
        "shard_meta.json",
        "faiss.index",
        "chunks.jsonl",
        "chunks.json",
        "chunks.offsets.npy",
        "bm25.pkl",
        "vectors.npy",
    ]:
        p = shard / name
        if p.exists():
            print(f"  {name:22} {p.stat().st_size / 1e9:.3f} GB")
    meta = shard / "shard_meta.json"
    if meta.exists():
        print("  meta:", meta.read_text(encoding="utf-8")[:400])
