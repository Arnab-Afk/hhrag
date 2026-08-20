"""Latency table from Downloads/benchmark.py, wired to this repo's HybridRetriever."""
from __future__ import annotations

import statistics
import sys
import time

from app.config import get_settings
from retrieval.hybrid import HybridRetriever

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    settings = get_settings()
    budget_ms = settings.retrieve_timeout_s * 1000

    print("Warming up (model load + first inference)...")
    retriever = HybridRetriever.load()
    for name in retriever.shards:
        retriever._shard(name)
    retriever.embedder.encode_query("warmup")
    retriever.retrieve(QUERIES[0], language_code="en", top_k=5)

    total_ms, embed_ms, search_ms = [], [], []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        query_vec = retriever.embedder.encode_query(query)
        e_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        shard = retriever._shard("en" if "en" in retriever.shards else next(iter(retriever.shards)))
        shard.search(
            query,
            query_vec,
            top_k=5,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        )
        s_ms = (time.perf_counter() - t1) * 1000
        embed_ms.append(e_ms)
        search_ms.append(s_ms)
        total_ms.append(e_ms + s_ms)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {budget_ms:.0f}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= budget_ms:
        print("PASS: within budget")
    else:
        print("FAIL: over budget")
        sys.exit(1)


if __name__ == "__main__":
    main()
