from pathlib import Path
import os
root = Path(r"D:\projects\hhrag")
env = root / ".env"
print("INDEX_DIR/HF lines:")
if env.exists():
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("INDEX_DIR", "LANGUAGES", "HF_", "TRANSFORMERS_")):
            key, _, val = line.partition("=")
            print(f"  {key}={val[:80]}")
print("os HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE"))
print("os TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE"))
idx = root / "indexes"
print("indexes exists", idx.exists(), "as jsonl", (idx / "as" / "chunks.jsonl").exists())
if (idx / "as" / "chunks.jsonl").exists():
    print("as jsonl GB", (idx / "as" / "chunks.jsonl").stat().st_size / 1e9)
