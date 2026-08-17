"""Index the full MSMARCO-XI train split on a large machine.

Downloads each language parquet (~3.7GB, Hub-cached), expands every
query's passages, exact-fold dedups, chunks, embeds, and writes FAISS
HNSW + BM25 shards. English passages are taken once (from Hindi) so
they are not duplicated 13×.

Telugu has no train file on the Hub; it is skipped.

Resume-friendly: a finished language (shard_meta.json) is skipped unless
you pass --overwrite.

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
from ingest.hf_source import TRAIN_LANGUAGES, iter_language_rows, parquet_row_count
from ingest.subsample import row_to_passages
from retrieval.embedder import Embedder


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

    builders: dict[str, IndexBuilder] = {}
    dedupers: dict[str, StreamingDeduper] = {}

    def builder(lang: str) -> IndexBuilder:
        if lang not in builders:
            builders[lang] = IndexBuilder(lang, out_dir, embedder, batch_size=batch_size)
        return builders[lang]

    def deduper(lang: str) -> StreamingDeduper:
        if lang not in dedupers:
            dedupers[lang] = StreamingDeduper()
        return dedupers[lang]

    for lang in indic:
        shard_meta = out_dir / lang / "shard_meta.json"
        if shard_meta.exists() and not overwrite:
            print(f"skip {lang}: {shard_meta} exists (pass --overwrite to rebuild)")
            continue
        print(f"\n=== {lang} train parquet ===")
        take_english = lang == english_source and (
            overwrite or not (out_dir / "en" / "shard_meta.json").exists()
        )
        total = parquet_row_count(lang, "train", cache_dir=cache_dir)
        print(f"  {total} queries in this split")
        n_rows = 0
        for row in tqdm(
            iter_language_rows(lang, "train", limit=None, cache_dir=cache_dir),
            desc=lang,
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

    for lang, bld in builders.items():
        dd = dedupers.get(lang)
        extra = f" kept={dd.kept} dropped_dup={dd.dropped}" if dd else ""
        print(f"finalizing {lang}{extra}")
        bld.close()


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Full MSMARCO-XI train index build")
    parser.add_argument(
        "--languages",
        default=",".join(TRAIN_LANGUAGES),
        help="Indic codes. English is extracted from Hindi (or the first listed). Default: all train languages.",
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
