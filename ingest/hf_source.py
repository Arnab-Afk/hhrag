"""Load one MSMARCO-XI language file from the Hub.

The dataset card still shows `load_dataset(..., "hi")`, but the published
repo is parquet-only with a single BuilderConfig named `default`. Language
is a filename, not a config:

    train/hintrain.parquet   (~3.7 GB)
    validation/hinval.parquet (~462 MB)
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

HF_REPO = "ai4bharat/MSMARCO-XI"

LANG_STEM = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}
TRAIN_MISSING = {"te"}
TRAIN_LANGUAGES = [code for code in LANG_STEM if code not in TRAIN_MISSING]


def parquet_path(language: str, split: str) -> str:
    if language not in LANG_STEM:
        raise ValueError(
            f"Unknown language '{language}'. Choose from: {', '.join(LANG_STEM)}"
        )
    if split == "train" and language in TRAIN_MISSING:
        raise FileNotFoundError(
            "MSMARCO-XI has no Telugu train parquet. Use --split validation for te."
        )
    stem = LANG_STEM[language]
    if split == "train":
        return f"train/{stem}train.parquet"
    if split == "validation":
        return f"validation/{stem}val.parquet"
    raise ValueError(f"Unknown split '{split}'")


def download_parquet(language: str, split: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    filename = parquet_path(language, split)
    extra = os.environ.get("MSMARCO_PARQUET_DIR")
    if extra:
        local = Path(extra) / filename
        if local.exists():
            print(f"  using local {local}")
            return str(local)
    print(f"  downloading hf://datasets/{HF_REPO}/{filename}")
    print("  (Hindi train is ~3.7GB; validation is ~462MB. Cached after the first run.)")
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def parquet_row_count(language: str, split: str, cache_dir: str | None = None) -> int:
    import pyarrow.parquet as pq

    local_path = download_parquet(language, split, cache_dir=cache_dir)
    return int(pq.ParquetFile(local_path).metadata.num_rows)


def iter_language_rows(
    language: str,
    split: str,
    limit: int | None = None,
    cache_dir: str | None = None,
) -> Iterator[dict]:
    """Download the language parquet (Hub cache) and yield rows in batches.

    Does not load the whole uncompressed table into RAM.
    """
    import pyarrow.parquet as pq

    local_path = download_parquet(language, split, cache_dir=cache_dir)
    parquet = pq.ParquetFile(local_path)
    yielded = 0
    for batch in parquet.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return
