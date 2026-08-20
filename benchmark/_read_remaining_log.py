from pathlib import Path
p = Path(r"D:\projects\hhrag\remaining.log")
print("exists", p.exists(), "size", p.stat().st_size if p.exists() else 0)
if p.exists():
    text = p.read_text(encoding="utf-8", errors="replace")
    print(text[-4000:])
