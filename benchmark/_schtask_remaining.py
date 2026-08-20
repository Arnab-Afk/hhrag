import subprocess
from pathlib import Path

bat = r"D:\projects\hhrag\run_remaining.bat"
create = [
    "schtasks",
    "/Create",
    "/TN",
    "hhrag_remaining",
    "/TR",
    bat,
    "/SC",
    "ONCE",
    "/ST",
    "23:59",
    "/F",
]
print(subprocess.check_output(create, text=True, errors="replace"))
run = ["schtasks", "/Run", "/TN", "hhrag_remaining"]
print(subprocess.check_output(run, text=True, errors="replace"))
print(subprocess.check_output(["schtasks", "/Query", "/TN", "hhrag_remaining", "/V", "/FO", "LIST"], text=True, errors="replace"))
