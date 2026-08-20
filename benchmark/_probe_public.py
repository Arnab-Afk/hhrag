import json
import time
import urllib.request
from pathlib import Path

base = "https://voiceraghhgoa.arnabbhowmik.in/api/ask"
root = Path(r"E:/projects/hhrag/benchmark/results")
files = ["_ask_goa.json", "_ask_taj.json", "_ask_gst.json", "_ask_upi.json"]
retrieves = []
for name in files:
    payload = (root / name).read_bytes()
    req = urllib.request.Request(
        base, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    wall = (time.perf_counter() - t0) * 1000
    r = float(body.get("retrieval_ms") or 0)
    retrieves.append(r)
    print(
        name,
        f"wall={wall:.0f}",
        f"retrieve={r}",
        f"under200={body.get('retrieval_under_200ms')}",
        f"e2e={body.get('end_to_end_ms')}",
    )
if retrieves:
    s = sorted(retrieves)
    print("retrieve_ms", "p50", s[len(s) // 2], "min", s[0], "max", s[-1])
