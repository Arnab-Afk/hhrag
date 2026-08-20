"""Use the GPU box network window to finish MSMARCO parquet copies.

Does not touch the running Assamese finalize job.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

from ingest.hf_source import HF_REPO, TRAIN_MISSING, parquet_path

ROOT = Path(r"D:\projects\hhrag")
DEST = ROOT / "msmarco"
LANGS = "as,bn,gu,kn,ml,mr,ne,or,pa,sa,ta,ur,te".split(",")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        split = "validation" if lang in TRAIN_MISSING else "train"
        rel = parquet_path(lang, split)
        dest = DEST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== {lang} {rel} ===", flush=True)
        url = hf_hub_url(HF_REPO, rel, repo_type="dataset")
        meta = get_hf_file_metadata(url)
        if dest.exists() and dest.stat().st_size == meta.size:
            print(f"  skip complete {dest.stat().st_size / 1e9:.2f} GB", flush=True)
            continue
        src = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                filename=rel,
                repo_type="dataset",
            )
        )
        print(f"  copy {src} -> {dest} ({meta.size / 1e9:.2f} GB)", flush=True)
        shutil.copy2(src, dest)
        print(f"  wrote {dest.stat().st_size / 1e9:.2f} GB", flush=True)
    print("prefetch done", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
