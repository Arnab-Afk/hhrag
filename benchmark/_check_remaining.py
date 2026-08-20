from pathlib import Path
import subprocess

for p in [
    Path(r"D:\projects\hhrag\remaining_heartbeat.txt"),
    Path(r"D:\projects\hhrag\remaining_inner.log"),
    Path(r"D:\projects\hhrag\remaining.log"),
]:
    print("==", p, "exists", p.exists(), "size", p.stat().st_size if p.exists() else 0)
    if p.exists():
        print(p.read_text(encoding="utf-8", errors="replace")[-2500:])

print("== schtasks")
print(subprocess.check_output(["schtasks", "/Query", "/TN", "hhrag_remaining", "/FO", "LIST"], text=True, errors="replace"))
