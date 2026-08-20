# HH RAG Goa — agent handoff

Read this before changing ingest, retrieval, deploy, or generation.
Laptop repo: `E:\projects\hhrag`. GitHub: `https://github.com/Arnab-Afk/hhrag.git` (often **behind** the machines).

**Hackathon:** HH Goa 2026 Task 2. Deadline **22 Aug 2026 23:59 IST**. One Google form, **no resubmits**: https://forms.gle/MNvCjcv23Hn2Eeu58  
**Live demo:** https://voiceraghhgoa.arnabbhowmik.in/

---

## What this system is

Voice RAG on [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):

Mic → Sarvam `saaras:v3` → hybrid retrieve (FAISS HNSW + BM25 **rerank on dense top 50**) → generate (Anthropic Haiku, extractive fallback) → guardrails.

The spec “200ms” is **retrieve only** (embed + search). STT and LLM are network hops; do not fake e2e 200ms.

---

## Machines (look here first)

| SSH host | What | Paths |
| --- | --- | --- |
| `swdc` | Linux jump + **production API**. User `swdc-kjsse`. LAN `172.17.1.111`. `/vm-storage` ~11TB. | App `/vm-storage/hhrag/src`, venv `/vm-storage/hhrag/venv`, live indexes `/vm-storage/hhrag/indexes`, parquet mirror `/vm-storage/hhrag/msmarco` |
| `swdc-win` | Windows **GPU ingest** (`admin@172.17.15.250`). RTX 5000 Ada. ProxyJump `swdc`. | Code + indexes `D:\projects\hhrag`, venv `D:\projects\hhrag\venv`, parquets `D:\projects\hhrag\msmarco` |
| Laptop | This git checkout. | `E:\projects\hhrag`. SSH config `C:\Users\arnab\.ssh\config` hosts `swdc` / `swdc-win`. Do **not** add LocalForward 7860. |

```
ssh swdc
ssh swdc-win
```

`swdc` SSH uses cloudflared Access. Nginx Proxy Manager runs in Docker `proxy-app-1` on **host network**. Public host **id 94**: `voiceraghhgoa.arnabbhowmik.in` → `127.0.0.1:7860`.

Production unit: `sudo systemctl restart hhrag` on `swdc`. Unit file: `deploy/hhrag.service` (`EnvironmentFile=/vm-storage/hhrag/src/.env`).

GPU box still has leftover uvicorn on 7860; **public traffic is swdc**, not Windows.

---

## Do not

- `load_dataset("ai4bharat/MSMARCO-XI", "hi")` — configs are filenames, e.g. `train/hintrain.parquet`. See `ingest/hf_source.py`.
- Corpus-wide `rank_bm25.get_scores` on millions of docs (was 18–33s retrieve). Runtime uses `rescore_bm25()` on the dense shortlist (`SPARSE_CANDIDATES = 50` in `retrieval/hybrid.py`).
- Load full `bm25.pkl` into the API. Live shards do not need it; copy `faiss.index`, `chunks.jsonl`, `chunks.offsets.npy`, `shard_meta.json`, `dense_meta.json` only.
- `--overwrite` full_train on `hi`/`en`/`as`/`bn`/`ta` while jsonl already exists.
- Commit `.env`, `.anthropic_key`, or API keys. An Anthropic key was pasted in chat — treat as exposed; rotate.
- Wait for Telugu **train** — Hub has none (`TRAIN_MISSING = {te}`). Use `validation/telval.parquet`.
- Warm every FAISS shard on API startup if many languages are loaded. Each HNSW index is ~12GB. `swdc` has ~251Gi RAM; `hi`+`en` already sit in production. Load on demand if adding 10+ shards.

---

## Live production (swdc) — last known good

- Shards serving: **`hi` + `en`** full train (not demo).
  - hi: 7,152,662 chunks, ~27GB
  - en: 6,879,259 chunks, ~18GB
- `.env`: `INDEX_DIR=/vm-storage/hhrag/indexes`, `LANGUAGES=hi,en`, Anthropic Haiku `claude-haiku-4-5-20251001`, Sarvam configured.
- Health: `GET http://127.0.0.1:7860/api/health` on swdc (`llm_configured` true).
- Retrieve ~30–70ms on full index. Generate ~1.8s (Haiku). Public wall-clock includes Cloudflare.
- Embedder: `intfloat/multilingual-e5-small` on **CPU** on swdc (`torch.cuda` false).
- Off-topic gate is weak (`MIN_RETRIEVAL_SCORE=0.012`); MSMARCO always hits something.

Laptop `python -m urllib` to the public URL can 403 (Cloudflare). Probe **localhost on swdc**.

---

## GPU ingest (swdc-win) — current job

**Goal:** finish every MSMARCO-XI language except missing Telugu train.

Scheduled task `hhrag_remaining` runs:

```
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.remaining --batch-size 512 --out D:\projects\hhrag\indexes
```

Log: `D:\projects\hhrag\remaining.log`  
Heartbeat: `D:\projects\hhrag\remaining_heartbeat.txt`  
Tracebacks: `D:\projects\hhrag\remaining_inner.log`  
Bat: `D:\projects\hhrag\run_remaining.bat`

Order in `ingest/remaining.py`:

1. **Finalize** leftover jsonl (no Hub): `as`, `bn`, `ta`
2. **full_train** from local parquets: `gu,kn,ml,mr,ne,or,pa,sa,ur` then `te` (validation)
3. Skip `hi`/`en` (already have `shard_meta.json`)
4. Skip rebuild if a large `chunks.jsonl` exists without meta — finalize instead

As of **21 Aug 2026 ~01:30 IST**: Assamese finalize was running (~6% of ~7.48M chunks, ~600–900 chunks/s, ~2–3h for `as` alone). Then `bn` and `ta` jsonl (~15GB and ~19GB). Then nine full-train languages (hours each). Total GPU time is **overnight / a day**, not minutes.

Windows TLS to huggingface.co often fails (`CERTIFICATE_VERIFY_FAILED` hostname mismatch / MITM). User may temporarily fix it. **Parquets are already on disk** — ingest must stay **offline**:

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
MSMARCO_PARQUET_DIR=D:\projects\hhrag\msmarco
```

`ingest/hf_source.py` uses `MSMARCO_PARQUET_DIR / train/<stem>train.parquet` before Hub.

Do **not** kill `ingest.remaining` to “fix” network. Finalize/embed is local GPU.

### Check ingest

```
ssh swdc-win
# log tail
type D:\projects\hhrag\remaining.log
schtasks /Query /TN hhrag_remaining
```

Finished language ⇒ `D:\projects\hhrag\indexes\<lang>\shard_meta.json` exists.

### Other Windows tasks

| Task | Purpose |
| --- | --- |
| `hhrag_remaining` | Main ingest (keep) |
| `hhrag_prefetch` | Hub parquet copy — **done** 21 Aug 01:33 |
| `hhrag_parquets` | LAN curl from swdc — stop if still running |
| `hhrag_idxhttp` | `python -m http.server 8766` serving `D:\projects\hhrag\indexes` |

---

## Parquets (already downloaded)

Hub repo `ai4bharat/MSMARCO-XI`. Stems in `ingest/hf_source.py` `LANG_STEM`.

Copies:

- swdc: `/vm-storage/hhrag/msmarco/train/*.parquet` + `validation/telval.parquet` (~43GB)
- win: `D:\projects\hhrag\msmarco\` (same layout, prefetch completed)

swdc can HTTP-serve that tree: `python3 -m http.server 8765 --directory /vm-storage/hhrag` (LAN `http://172.17.1.111:8765/msmarco/...`).

---

## Copy a finished shard swdc ← GPU

Windows inbound **8766** was opened (`netsh advfirewall ... localport=8766`). From swdc:

```
bash /vm-storage/hhrag/src/ingest/pull_shard.sh as
sudo systemctl restart hhrag
curl -sS http://127.0.0.1:7860/api/health
```

Script: `ingest/pull_shard.sh` (curl `http://172.17.15.250:8766/<lang>/...`).

`ssh admin@172.17.15.250` from swdc is **publickey denied**. Win→swdc `scp` also failed (no key). Use HTTP, not scp, unless keys are added.

After pull, `HybridRetriever.load()` scans `INDEX_DIR` children with `shard_meta.json` — you do not have to list every language in `LANGUAGES` for them to load, but lifespan currently **warms all shards**. Watch RAM before adding 10 FAISS indexes.

---

## Code map (what matters)

| Path | Role |
| --- | --- |
| `app/harness.py` | Pipeline, timings, LLM vs extractive |
| `app/config.py` | Settings; relative `INDEX_DIR` resolved under repo ROOT |
| `generation/llm.py` | Anthropic Haiku first, then Groq, then extractive |
| `retrieval/hybrid.py` | Language-routed FAISS + shortlist BM25 |
| `retrieval/chunk_store.py` | `chunks.offsets.npy` — seek jsonl by FAISS id |
| `retrieval/sparse.py` | `rescore_bm25`; `SparseIndex.save/load` on the class (not nested) |
| `ingest/full_train.py` | **Close each language before the next** |
| `ingest/finalize_jsonl.py` | Embed jsonl → FAISS + offsets; skip corpus BM25 pickle |
| `ingest/remaining.py` | Orchestrator on the GPU box |
| `ingest/prefetch_parquets.py` | One-shot Hub copy into `msmarco/` |
| `ingest/download_parquets.py` | Same idea on Linux |
| `frontend/` | Mic UI |
| `data/demo/` | Tiny fixture corpus + queries |

`IndexBuilder` default `skip_sparse=True` (no `bm25.pkl`).

---

## How to work from another agent checkout

1. Open this repo; SSH `swdc` / `swdc-win` as above (need `id_rsa` + cloudflared).
2. Status: `remaining.log` on win; `systemctl status hhrag` and `ls /vm-storage/hhrag/indexes` on swdc.
3. If `indexes/<lang>/shard_meta.json` appeared on win, `pull_shard.sh` then restart `hhrag`.
4. If ingest died: **do not overwrite** large jsonl. Re-run `python -m ingest.remaining` (skips finished meta; finalizes leftover jsonl).
5. Laptop git is missing production fixes until someone commits: offsets, BM25 rescore, Anthropic, `ingest/remaining.py`, `chunk_store.py`, `deploy/`.
6. Tests: `python -m pytest` (retrieval offsets + rescore). Latency: `python -m benchmark.run_latency` against a **real** index, not demo 13-chunks.
7. Submission still needs: latency table in README/form, videos + `#RAGInGoa`, push GitHub, form once.

---

## Keys (locations only)

- swdc `/vm-storage/hhrag/src/.env` — Sarvam, Anthropic. Gitignored.
- win `D:\projects\hhrag\.env` — `INDEX_DIR=indexes` (relative; code resolves to `D:\projects\hhrag\indexes`).
- Never echo keys. Rotate Anthropic if it was pasted in a chat.

---

## Snapshot

Written **21 Aug 2026 ~01:38 IST**.

- Production: hi+en RAG + Anthropic + voice, retrieve under 200ms.
- GPU: `ingest.remaining` embedding Assamese; parquets for all remaining langs **on disk**.
- Next agent job: watch `remaining.log` → pull each new shard to swdc → restart API → keep RAM in mind → commit when asked.
