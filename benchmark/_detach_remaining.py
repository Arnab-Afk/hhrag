CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

import os
import subprocess
from pathlib import Path

root = Path(r"D:\projects\hhrag")
env = os.environ.copy()
env.pop("HF_HUB_OFFLINE", None)
env.pop("TRANSFORMERS_OFFLINE", None)
env["HF_HUB_DISABLE_TELEMETRY"] = "1"
env["PYTHONUNBUFFERED"] = "1"
env["MSMARCO_PARQUET_DIR"] = str(root / "msmarco")
log = (root / "remaining.log").open("a", encoding="utf-8")
log.write("LAUNCH remaining --out indexes\n")
log.flush()
proc = subprocess.Popen(
    [
        str(root / "venv" / "Scripts" / "python.exe"),
        "-u",
        "-m",
        "ingest.remaining",
        "--batch-size",
        "512",
        "--out",
        str(root / "indexes"),
    ],
    cwd=str(root),
    stdout=log,
    stderr=subprocess.STDOUT,
    env=env,
    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
)
print("pid", proc.pid)
