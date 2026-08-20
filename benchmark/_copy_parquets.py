import subprocess
from pathlib import Path

log = Path(r"D:\projects\hhrag\parquet_copy.log")
cmd = [
    "scp",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-r",
    "swdc-kjsse@172.17.1.111:/vm-storage/hhrag/msmarco",
    r"D:\projects\hhrag\msmarco",
]
with log.open("w", encoding="utf-8") as handle:
    handle.write(" ".join(cmd) + "\n")
    handle.flush()
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
print("copy pid", proc.pid)
