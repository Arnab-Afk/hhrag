"""Hit a live /api/ask and print retrieve vs wall-clock latency."""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

QUERIES = [
    ("Where is the Taj Mahal and who built it?", "en-IN"),
    ("ताज महल कहाँ है?", "hi-IN"),
    ("When did GST start in India?", "en-IN"),
    ("भारत में GST कब लागू हुआ?", "hi-IN"),
    ("Who chaired the drafting committee of the Indian Constitution?", "en-IN"),
    ("Where does the Ganga originate?", "en-IN"),
    ("How many overs are there in a T20 match?", "en-IN"),
    ("What is UPI and who developed it?", "en-IN"),
    ("Where is ISRO headquartered?", "en-IN"),
    ("What is the capital of Goa?", "en-IN"),
    ("गोवा की राजधानी क्या है?", "hi-IN"),
    ("When was Article 370 abrogated?", "en-IN"),
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if p >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def ask(base: str, query: str, language: str, timeout: int = 180) -> dict:
    payload = json.dumps(
        {"query": query, "language": language, "mode": "extractive"}
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    wall_ms = (time.perf_counter() - t0) * 1000
    body["_wall_ms"] = wall_ms
    return body


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
    print(f"base={base}")
    print("warmup...")
    try:
        warm = ask(base, "What is the capital of India?", "en-IN")
        print(
            f"warmup wall={warm['_wall_ms']:.1f}ms retrieve={warm.get('retrieval_ms')} "
            f"e2e={warm.get('end_to_end_ms')} under200={warm.get('retrieval_under_200ms')} "
            f"refused={warm.get('refused')} mode={warm.get('generation_mode')}"
        )
    except Exception as exc:
        print(f"warmup failed: {exc}")
        raise

    walls: list[float] = []
    retrieves: list[float] = []
    e2es: list[float] = []
    for i, (query, lang) in enumerate(QUERIES, 1):
        try:
            row = ask(base, query, lang)
        except Exception as exc:
            print(f"{i:02d} FAIL {exc}")
            continue
        walls.append(row["_wall_ms"])
        retrieves.append(float(row.get("retrieval_ms") or 0))
        e2es.append(float(row.get("end_to_end_ms") or 0))
        snippet = (row.get("answer") or "")[:80].replace("\n", " ")
        print(
            f"{i:02d} wall={row['_wall_ms']:8.1f} retrieve={row.get('retrieval_ms'):8} "
            f"e2e={row.get('end_to_end_ms'):8} under200={str(row.get('retrieval_under_200ms')):5} "
            f"refused={row.get('refused')} shards={row.get('citations') and 'yes'} {snippet}"
        )

    def dump(name: str, values: list[float]) -> None:
        if not values:
            print(f"{name}: no samples")
            return
        print(
            f"{name} n={len(values)} p50={percentile(values,50):.1f} "
            f"p70={percentile(values,70):.1f} p100={percentile(values,100):.1f} "
            f"mean={statistics.fmean(values):.1f}"
        )

    dump("wall_ms", walls)
    dump("retrieve_ms", retrieves)
    dump("pipeline_e2e_ms", e2es)


if __name__ == "__main__":
    main()
