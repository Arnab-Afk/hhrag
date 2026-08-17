"""Measure P50 / P70 / P100 per stage over a query set.

The <200ms target is evaluated on retrieve (embed + hybrid search +
guardrail), not STT or LLM generation. Those network stages are timed
separately when keys are present.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
from pathlib import Path

from app.config import ROOT
from app.harness import Pipeline
from app.timing import Timer
from retrieval.hybrid import HybridRetriever


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if p >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(name: str, values: list[float]) -> dict:
    return {
        "stage": name,
        "n": len(values),
        "p50_ms": round(percentile(values, 50), 3),
        "p70_ms": round(percentile(values, 70), 3),
        "p100_ms": round(percentile(values, 100), 3),
        "mean_ms": round(statistics.fmean(values), 3) if values else 0.0,
    }


async def run_async(queries: list[dict], mode: str) -> list[dict]:
    pipe = Pipeline()
    rows: list[dict] = []
    for i, item in enumerate(queries, start=1):
        result = await pipe.run(
            query=item["query"],
            language=item.get("language"),
            mode=mode,  # type: ignore[arg-type]
        )
        by_stage = {t.name: t.ms for t in result.timings_ms}
        rows.append(
            {
                "i": i,
                "query": item["query"],
                "tag": item.get("tag", ""),
                "refused": result.refused,
                "retrieve_ms": by_stage.get("retrieve", 0.0),
                "generate_ms": by_stage.get("generate", 0.0),
                "e2e_ms": result.end_to_end_ms,
                "grounded": result.grounded,
            }
        )
        print(
            f"[{i:02d}] retrieve={by_stage.get('retrieve', 0):7.2f}ms  "
            f"e2e={result.end_to_end_ms:7.2f}ms  refused={result.refused}  {item['query'][:48]}"
        )
    return rows


def run_retrieve_only(queries: list[dict]) -> list[float]:
    retriever = HybridRetriever.load()
    times: list[float] = []
    # Warmup
    retriever.retrieve(queries[0]["query"], language_code=queries[0].get("language"))
    for item in queries:
        timer = Timer()
        with timer.stage("retrieve"):
            retriever.retrieve(item["query"], language_code=item.get("language"))
        times.append(timer.get("retrieve"))
    return times


def write_outputs(rows: list[dict], retrieve_times: list[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = [
        summarize("retrieve", retrieve_times or [r["retrieve_ms"] for r in rows]),
        summarize("generate", [r["generate_ms"] for r in rows if r["generate_ms"]]),
        summarize("end_to_end", [r["e2e_ms"] for r in rows]),
    ]
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    lines = [
        "| Stage | n | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        if s["n"] == 0:
            continue
        lines.append(
            f"| {s['stage']} | {s['n']} | {s['p50_ms']} | {s['p70_ms']} | {s['p100_ms']} | {s['mean_ms']} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "demo" / "queries.json")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark" / "results")
    parser.add_argument("--mode", default="extractive", choices=["auto", "extractive", "generative"])
    args = parser.parse_args()
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    retrieve_times = run_retrieve_only(queries)
    rows = asyncio.run(run_async(queries, args.mode))
    write_outputs(rows, retrieve_times, args.out)


if __name__ == "__main__":
    main()
