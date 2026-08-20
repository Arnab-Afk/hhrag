import runpy
import sys
from pathlib import Path

root = Path(r"D:\projects\hhrag")
sys.path.insert(0, str(root))
print("hello from wrapper", flush=True)
sys.argv = ["ingest.remaining", "--batch-size", "512"]
runpy.run_module("ingest.remaining", run_name="__main__")
