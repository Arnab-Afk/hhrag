---
title: HH RAG Goa
emoji: 🎙️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Voice-enabled RAG — HH Goa 2026 Task 2

Mic → **Sarvam saaras:v3** → hybrid retrieval (FAISS/numpy + BM25 → RRF) → constrained generation → guardrails.

Dataset: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (11.5M rows, 14 Indic languages). This repo does **not** index all of it. Default path: Hindi + English, subsampled, plus a bundled demo corpus so the pipeline runs before the first 50k-row download.

## Latency (read this before judging the 200ms line)

The spec’s “full process under 200ms” is not physically achievable if it includes **Sarvam STT** and **hosted LLM generation**. A Sarvam round-trip is typically hundreds of milliseconds to seconds; Groq is fast but still a network hop.

We **do not fake an end-to-end number**. We decompose:

| Stage | What we measure | Target |
| --- | --- | --- |
| STT | Sarvam `/speech-to-text` | reported honestly (not in the 200ms budget) |
| **retrieve** | query embed + dense ANN + BM25 + RRF + parent-child assemble | **&lt; 200ms** (this is the controllable stage) |
| generate | Groq JSON completion, or extractive fallback | reported separately (TTFT / full) |
| guardrails | input / retrieval-gate / grounding | included in retrieve or as their own rows |

Run:

```powershell
python -m ingest.build_index --demo
python -m benchmark.run_latency
```

That writes `benchmark/results/summary.md` with **P50 / P70 / P100** over the demo query set (on-topic, numeric, off-topic, unsafe). Paste those numbers into the submission form. P100 is the **max**, not a marketing percentile.

## Chunking (not one naive window)

MSMARCO passages are already units. We still run a stacked strategy because translations vary in length and the corpus is full of near-duplicates.

1. **Passage-native** — short passages stay whole, tagged with `query_id`, `query_type`, `is_selected`, `source` (translated vs English), language.
2. **Sentence packing + overlap** — script-aware split (Devanagari danda `।`, protected decimals/abbreviations), then 128-token windows with 24-token overlap.
3. **Semantic / adaptive** — consecutive-sentence character-ngram cosine; break on topic shift or length.
4. **Parent-child** — embed the small unit, return the **full parent passage** to the generator.
5. **Simhash dedup** — Hamming ≤ 3 drops near-duplicate passages before indexing.
6. **Metadata-aware retrieval** — `query_type` boost (NUMERIC/PERSON/LOCATION) at fusion time. **`is_selected` is stored and never used as a ranking signal** (that would leak MSMARCO labels).

See `chunking/pipeline.py`.

## Architecture

```
Mic → Sarvam STT (saaras:v3, language_code=unknown)
    → Guardrail: empty / garbage / unsafe
    → Query embed (local multilingual-e5-small)
    → Language-routed hybrid search
         dense: FAISS HNSW (numpy cosine fallback on Windows)
         sparse: BM25 per shard
         fuse: reciprocal rank fusion + query_type boost
    → Guardrail: min RRF score (no generator call if nothing clears it)
    → Generate: Groq JSON, or extractive top-passage if no key
    → Guardrail: citation IDs must exist; lexical/ngram overlap vs context
    → Structured PipelineResponse
```

Harness (`app/harness.py`): pydantic I/O, per-stage `perf_counter`, STT/LLM retries with backoff, explicit refusal strings instead of crashing.

## Quick start (Windows / PowerShell)

```powershell
cd e:\projects\hhrag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
copy .env.example .env
# put SARVAM_API_KEY and optional GROQ_API_KEY in .env

python -m ingest.build_index --demo
python -m pytest
uvicorn app.main:app --reload --port 7860
```

Open http://127.0.0.1:7860

Without `GROQ_API_KEY` the system answers **extractively** (top parent passage). That is a valid grounded mode and the right way to demo the 200ms retrieval budget. Generative mode needs Groq.

Without `SARVAM_API_KEY` you can still type queries. Voice needs Sarvam.

### Index a real MSMARCO-XI subsample

```powershell
python -m ingest.subsample --languages hi,en --max-queries 8000 --split train
python -m ingest.build_index --languages hi,en
```

8000 Hindi queries ≈ tens of thousands of passages after expanding `Translated_passages` / `English_passages` and running dedup. Raise `--max-queries` toward 30k–80k when you have overnight indexing time. Add `ta` or `bn` the same way.

Use `--split validation` and hold those queries out of the train index for the latency/quality notebook.

## API

- `GET /api/health`
- `POST /api/ask` `{ "query": "...", "mode": "auto"|"extractive"|"generative" }`
- `POST /api/ask/voice` multipart `file` + `mode`

Response always includes `refused`, `guardrails[]`, `timings_ms[]`, `retrieval_ms`, `retrieval_under_200ms`.

## Guardrails (when not to answer)

| Gate | Behaviour |
| --- | --- |
| Input | Reject empty, punctuation-only, repeated-character garbage, basic unsafe patterns |
| Retrieval | If max fused score &lt; `MIN_RETRIEVAL_SCORE`, **do not call the LLM** — return *I don't have information on that in this dataset.* |
| Grounding | Citations must refer to retrieved chunks; answer–context overlap must clear a threshold or we refuse rather than serve the ungrounded text |

Off-topic demo queries in `data/demo/queries.json` are there so the refusal path shows up in the demo video.

## Stack

- FastAPI + pydantic-settings
- Sarvam `saaras:v3` REST (`language_code=unknown` → shard routing)
- `intfloat/multilingual-e5-small` locally via sentence-transformers
- FAISS HNSW inner-product on normalized vectors; **numpy fallback** if `faiss-cpu` has no Windows wheel
- `rank_bm25` + RRF (`k=60`)
- Groq `llama-3.1-8b-instant` JSON mode (optional)
- Vanilla HTML/CSS/JS frontend (`MediaRecorder`)

## Repo map

```
app/            FastAPI, harness, schemas, timing
stt/            Sarvam client (retries)
chunking/       script-aware, overlap, semantic, dedup, parent-child
retrieval/      embedder, dense, BM25, RRF, language routing
guardrails/     input, retrieval gate, grounding
generation/     constrained JSON prompt + extractive fallback
ingest/         MSMARCO-XI streaming subsample + index build
benchmark/      P50/P70/P100 runner
frontend/       inquiry-desk UI
data/demo/      bilingual fixture corpus + queries
```

## Submission checklist

- [ ] Real MSMARCO-XI shards (not only demo) for the live link
- [ ] `python -m benchmark.run_latency` numbers pasted into README + form
- [ ] Live URL (Hugging Face Spaces Docker is wired via this README’s YAML + `Dockerfile`)
- [ ] GitHub repo public
- [ ] Video 1 process (90s) + Video 2 e2e demo
- [ ] Every teammate posts both videos on Instagram, X, LinkedIn with **`#RAGInGoa`**
- [ ] Form: https://forms.gle/MNvCjcv23Hn2Eeu58 — **no resubmissions**, deadline 22 Aug 2026 23:59

## Keys

- Sarvam: https://dashboard.sarvam.ai — header is `api-subscription-key`, not Bearer
- Groq: https://console.groq.com
