from pathlib import Path
p = Path(r"D:\projects\hhrag\prefetch.log")
print("exists", p.exists(), "size", p.stat().st_size if p.exists() else 0)
if p.exists():
    print(p.read_text(encoding="utf-8", errors="replace")[-3000:])
