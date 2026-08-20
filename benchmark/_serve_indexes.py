CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
import subprocess
from pathlib import Path

log = Path(r"D:\projects\hhrag\index_http.log").open("a", encoding="utf-8")
proc = subprocess.Popen(
    [
        r"D:\projects\hhrag\venv\Scripts\python.exe",
        "-m",
        "http.server",
        "8766",
        "--bind",
        "0.0.0.0",
        "--directory",
        r"D:\projects\hhrag\indexes",
    ],
    cwd=r"D:\projects\hhrag",
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
)
print("index http pid", proc.pid)
