"""Index the full MSMARCO-XI train split on a large machine.

Downloads each language parquet (~3.7GB, Hub-cached), expands every
query's passages, exact-fold dedups, chunks, embeds, and writes FAISS
HNSW shards. English passages are taken once (from Hindi) so they are
not duplicated 13×.

Telugu has no train file on the Hub; we index the validation split
instead so a `te` shard still exists.

Resume-friendly: a finished language (shard_meta.json) is skipped unless
you pass --overwrite. Languages with a large chunks.jsonl but no meta
are left for `python -m ingest.finalize_jsonl` so we do not wipe them.

Each language is closed before the next starts so a crash cannot lose
every in-memory FAISS index at once.

Example (the big VM):

    python -m ingest.full_train
    python -m ingest.full_train --languages hi,en
    python -m ingest.full_train --overwrite --languages ta
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from app.config import get_settings
from chunking.dedup import StreamingDeduper
from ingest.build_index import IndexBuilder
from ingest.hf_source import TRAIN_LANGUAGES, TRAIN_MISSING, iter_language_rows, parquet_row_count
from ingest.subsample import row_to_passages
from retrieval.embedder import Embedder

JSONL_KEEP_BYTES = 50_000_000


def _split_for(lang: str) -> str:
    return "validation" if lang in TRAIN_MISSING else "train"


def _should_skip(lang: str, out_dir: Path, overwrite: bool) -> bool:
    shard_meta = out_dir / lang / "shard_meta.json"
    jsonl = out_dir / lang / "chunks.jsonl"
    if shard_meta.exists() and not overwrite:
        print(f"skip {lang}: {shard_meta} exists (pass --overwrite to rebuild)")
        return True
    if (
        not overwrite
        and jsonl.exists()
        and jsonl.stat().st_size > JSONL_KEEP_BYTES
        and not shard_meta.exists()
    ):
        print(
            f"skip {lang}: {jsonl} is {jsonl.stat().st_size / 1e9:.2f} GB without "
            "shard_meta — run python -m ingest.finalize_jsonl --languages "
            f"{lang} instead of rebuilding"
        )
        return True
    return False


def index_language(
    lang: str,
    out_dir: Path,
    embedder: Embedder,
    batch_size: int,
    take_english: bool,
    cache_dir: str | None,
) -> None:
    split = _split_for(lang)
    if split != "train":
        print(f"  {lang}: Hub has no train parquet; using {split}")
    print(f"\n=== {lang} {split} parquet ===")
    total = parquet_row_count(lang, split, cache_dir=cache_dir)
    print(f"  {total} queries in this split")
    builders: dict[str, IndexBuilder] = {}
    dedupers: dict[str, StreamingDeduper] = {}

    def builder(code: str) -> IndexBuilder:
        if code not in builders:
            builders[code] = IndexBuilder(
                code, out_dir, embedder, batch_size=batch_size, skip_sparse=True
            )
        return builders[code]

    def deduper(code: str) -> StreamingDeduper:
        if code not in dedupers:
            dedupers[code] = StreamingDeduper()
        return dedupers[code]

    n_rows = 0
    for row in tqdm(
        iter_language_rows(lang, split, limit=None, cache_dir=cache_dir),
        desc=f"{lang}/{split}",
        total=total,
        mininterval=2.0,
    ):
        n_rows += 1
        for passage in row_to_passages(
            row,
            lang,
            include_english=take_english,
            selected_only=False,
        ):
            if not deduper(passage.language).keep(passage.text):
                continue
            builder(passage.language).add_passage(passage)
    print(f"  {lang}: read {n_rows} queries")
    for code, bld in builders.items():
        dd = dedupers.get(code)
        extra = f" kept={dd.kept} dropped_dup={dd.dropped}" if dd else ""
        print(f"finalizing {code}{extra}")
        bld.close()


def run(
    languages: list[str],
    out_dir: Path,
    batch_size: int,
    overwrite: bool,
    cache_dir: str | None,
) -> None:
    indic = [lang for lang in languages if lang != "en"]
    if not indic:
        raise SystemExit("Need at least one Indic language (English is taken from that parquet).")
    english_source = "hi" if "hi" in indic else indic[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    print(f"Embedding model: {embedder.model_name}")
    print(f"Languages: {indic}  |  English from: {english_source}")

    for lang in indic:
        if _should_skip(lang, out_dir, overwrite):
            continue
        take_english = lang == english_source and (
            overwrite or not (out_dir / "en" / "shard_meta.json").exists()
        )
        index_language(lang, out_dir, embedder, batch_size, take_english, cache_dir)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Full MSMARCO-XI train index build")
    parser.add_argument(
        "--languages",
        default=",".join(TRAIN_LANGUAGES + sorted(TRAIN_MISSING)),
        help="Language codes. Telugu uses validation (no train on Hub). "
        "English is extracted from Hindi (or the first listed).",
    )
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    run(langs, args.out, args.batch_size, args.overwrite, args.cache_dir)


if __name__ == "__main__":
    main()
