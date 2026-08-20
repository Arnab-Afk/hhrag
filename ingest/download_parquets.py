"""Pre-download MSMARCO-XI parquets (Linux box, then copy to the GPU machine).

    python -m ingest.download_parquets --out /vm-storage/hhrag/msmarco
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ingest.hf_source import TRAIN_MISSING, download_parquet, parquet_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--languages",
        default="as,bn,gu,kn,ml,mr,ne,or,pa,sa,ta,ur,te",
        help="Skip hi; Hindi parquet is already cached on the GPU box.",
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        split = "validation" if lang in TRAIN_MISSING else "train"
        src = Path(download_parquet(lang, split, cache_dir=args.cache_dir))
        dest = args.out / parquet_path(lang, split)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            print(f"already have {dest}")
            continue
        print(f"copy {src} -> {dest}")
        shutil.copy2(src, dest)
        print(f"  {dest.stat().st_size / 1e9:.2f} GB")
    print("done", args.out)


if __name__ == "__main__":
    main()
