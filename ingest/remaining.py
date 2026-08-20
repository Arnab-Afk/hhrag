"""Finish remaining MSMARCO-XI shards: finalize leftover jsonl, then index the rest.

Safe to re-run. Does not overwrite hi/en. Telugu uses the Hub validation split.

    python -m ingest.remaining
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.config import ROOT, get_settings
from ingest.finalize_jsonl import finalize_language
from ingest.full_train import run as full_train_run
from ingest.hf_source import TRAIN_LANGUAGES, TRAIN_MISSING
from retrieval.embedder import Embedder

# Already on disk as chunks.jsonl without shard_meta (Windows GPU box).
FINALIZE_FIRST = ["as", "bn", "ta"]
DONE = {"hi", "en"}


def remaining_codes() -> list[str]:
    codes = [c for c in TRAIN_LANGUAGES if c not in DONE and c not in FINALIZE_FIRST]
    codes.extend(sorted(TRAIN_MISSING))
    return codes


def main() -> None:
    log_path = Path("remaining_inner.log")
    try:
        _main_impl()
    except BaseException:
        import traceback

        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _main_impl() -> None:
    os.chdir(ROOT)
    # GPU box has a TLS MITM intercept. Stay offline for Hub; use local
    # model cache + MSMARCO_PARQUET_DIR copies from swdc.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--flush-every", type=int, default=2048)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Skip as/bn/ta jsonl finalize (only run full_train for missing langs).",
    )
    args = parser.parse_args()
    print(f"ROOT={ROOT} out={args.out.resolve()}", flush=True)

    embedder = Embedder()
    print(f"Embedding model: {embedder.model_name} dim={embedder.dim}")

    if not args.skip_finalize:
        for lang in FINALIZE_FIRST:
            jsonl = args.out / lang / "chunks.jsonl"
            if not jsonl.exists():
                print(f"no jsonl for {lang}; will full-train later")
                continue
            print(f"\n>>> finalize leftover {lang}")
            finalize_language(
                lang,
                args.out,
                embedder,
                args.batch_size,
                args.flush_every,
                overwrite=False,
            )

    extra = [c for c in FINALIZE_FIRST if not (args.out / c / "shard_meta.json").exists()]
    langs = remaining_codes() + extra
    print(f"\n>>> full_train {langs}")
    full_train_run(
        langs,
        args.out,
        args.batch_size,
        overwrite=False,
        cache_dir=args.cache_dir,
    )
    print("remaining ingest done")


if __name__ == "__main__":
    Path(r"D:\projects\hhrag\remaining_heartbeat.txt").write_text("started\n", encoding="utf-8")
    main()
