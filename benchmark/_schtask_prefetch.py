import subprocess
from pathlib import Path

bat = r"D:\projects\hhrag\run_prefetch.bat"
create = [
    "schtasks", "/Create", "/TN", "hhrag_prefetch",
    "/TR", bat, "/SC", "ONCE", "/ST", "23:57", "/F",
]
print(subprocess.check_output(create, text=True, errors="replace"))
print(subprocess.check_output(["schtasks", "/Run", "/TN", "hhrag_prefetch"], text=True, errors="replace"))
print("prefetch task started")
