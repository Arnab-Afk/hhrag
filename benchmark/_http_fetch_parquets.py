import subprocess
from pathlib import Path

# Stop hung scp if still running
subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     "Get-Process scp -ErrorAction SilentlyContinue | Stop-Process -Force"],
    check=False,
)

base = "http://172.17.1.111:8765/msmarco"
dest = Path(r"D:\projects\hhrag\msmarco")
(dest / "train").mkdir(parents=True, exist_ok=True)
(dest / "validation").mkdir(parents=True, exist_ok=True)
files = [
    ("train/asmtrain.parquet", dest / "train" / "asmtrain.parquet"),
    ("train/bentrain.parquet", dest / "train" / "bentrain.parquet"),
    ("train/gujtrain.parquet", dest / "train" / "gujtrain.parquet"),
    ("train/kantrain.parquet", dest / "train" / "kantrain.parquet"),
    ("train/maltrain.parquet", dest / "train" / "maltrain.parquet"),
    ("train/martrain.parquet", dest / "train" / "martrain.parquet"),
    ("train/neptrain.parquet", dest / "train" / "neptrain.parquet"),
    ("train/oritrain.parquet", dest / "train" / "oritrain.parquet"),
    ("train/pantrain.parquet", dest / "train" / "pantrain.parquet"),
    ("train/santrain.parquet", dest / "train" / "santrain.parquet"),
    ("train/tamtrain.parquet", dest / "train" / "tamtrain.parquet"),
    ("train/urdtrain.parquet", dest / "train" / "urdtrain.parquet"),
    ("validation/telval.parquet", dest / "validation" / "telval.parquet"),
]
log = Path(r"D:\projects\hhrag\parquet_copy.log")
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
handle = log.open("a", encoding="utf-8")
handle.write("HTTP copy start\n")
handle.flush()
# curl.exe is on modern Windows
cmd = ["cmd", "/c"]
lines = []
for rel, path in files:
    url = f"{base}/{rel}"
    lines.append(f'curl.exe -L --retry 5 --retry-all-errors -C - -o "{path}" "{url}"')
script = dest / "fetch.bat"
script.write_text("\n".join(lines) + "\n", encoding="utf-8")
proc = subprocess.Popen(
    ["cmd", "/c", str(script)],
    cwd=str(dest),
    stdout=handle,
    stderr=subprocess.STDOUT,
    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
)
print("fetch pid", proc.pid, "bat", script)
