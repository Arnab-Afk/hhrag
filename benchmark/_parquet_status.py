from pathlib import Path
import os

root = Path(r"D:\projects\hhrag\msmarco")
print("msmarco exists", root.exists())
total = 0
if root.exists():
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            print(p, round(p.stat().st_size / 1e9, 2), "GB")
print("total GB", round(total / 1e9, 2))
log = Path(r"D:\projects\hhrag\parquet_copy.log")
print("copy log size", log.stat().st_size if log.exists() else 0)
print(log.read_text(encoding="utf-8", errors="replace")[-500:] if log.exists() else "")
