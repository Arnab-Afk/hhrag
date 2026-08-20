import subprocess
from pathlib import Path

bat = Path(r"D:\projects\hhrag\msmarco\fetch.bat")
print("bat exists", bat.exists())
if bat.exists():
    print(bat.read_text(encoding="utf-8")[:800])
create = [
    "schtasks", "/Create", "/TN", "hhrag_parquets",
    "/TR", str(bat), "/SC", "ONCE", "/ST", "23:58", "/F",
]
print(subprocess.check_output(create, text=True, errors="replace"))
print(subprocess.check_output(["schtasks", "/Run", "/TN", "hhrag_parquets"], text=True, errors="replace"))
