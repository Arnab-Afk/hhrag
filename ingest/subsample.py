from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from app.config import ROOT, get_settings
from chunking.types import RawPassage

LANG_FILES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def row_to_passages(
    row: dict,
    language: str,
    include_english: bool,
    selected_only: bool,
) -> list[RawPassage]:
    passages = row.get("passages") or {}
    selected = _as_list(passages.get("is_selected"))
    translated = _as_list(passages.get("Translated_passages"))
    english = _as_list(passages.get("English_passages"))
    query_id = int(row.get("query_id") or 0)
    query_type = row.get("query_type")
    out: list[RawPassage] = []
    n = max(len(translated), len(english), len(selected))
    for i in range(n):
        flag = int(selected[i]) if i < len(selected) else 0
        if selected_only and flag != 1:
            continue
        if i < len(translated) and translated[i]:
            out.append(
                RawPassage(
                    query_id=query_id,
                    query_type=query_type,
                    language=language,
                    passage_index=i,
                    is_selected=flag,
                    source="translated",
                    text=translated[i],
                )
            )
        if include_english and i < len(english) and english[i]:
            out.append(
                RawPassage(
                    query_id=query_id,
                    query_type=query_type,
                    language="en",
                    passage_index=i,
                    is_selected=flag,
                    source="english",
                    text=english[i],
                )
            )
    return out


def subsample(
    languages: list[str],
    max_queries: int,
    split: str,
    selected_ratio: float,
    out_dir: Path,
) -> dict[str, list[RawPassage]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_lang: dict[str, list[RawPassage]] = defaultdict(list)
    indic = [lang for lang in languages if lang != "en"]
    english_source = indic[0] if indic else None

    for lang in languages:
        if lang == "en":
            continue
        print(f"Streaming {lang} / {split} (max_queries={max_queries})...")
        ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=split, streaming=True)
        seen = 0
        selected_budget = int(max_queries * selected_ratio)
        other_budget = max_queries - selected_budget
        n_sel = n_other = 0
        type_counts: dict[str, int] = defaultdict(int)

        for row in tqdm(ds, total=max_queries, desc=lang):
            qtype = row.get("query_type") or "DESCRIPTION"
            # Light stratification: cap any one type at 45% of the budget.
            if type_counts[qtype] >= int(max_queries * 0.45) and seen > 20:
                continue
            passages = row.get("passages") or {}
            flags = _as_list(passages.get("is_selected"))
            has_selected = any(int(x) == 1 for x in flags)
            if has_selected:
                if n_sel >= selected_budget:
                    continue
                n_sel += 1
            else:
                if n_other >= other_budget:
                    continue
                n_other += 1
            type_counts[qtype] += 1
            include_en = lang == english_source
            for passage in row_to_passages(
                row, lang, include_english=include_en, selected_only=False
            ):
                by_lang[passage.language].append(passage)
            seen += 1
            if seen >= max_queries:
                break

        print(f"  {lang}: {seen} queries, selected={n_sel} other={n_other} types={dict(type_counts)}")

    for lang, passages in by_lang.items():
        dest = out_dir / f"{lang}_{split}.jsonl"
        with dest.open("w", encoding="utf-8") as handle:
            for p in passages:
                handle.write(
                    json.dumps(
                        {
                            "query_id": p.query_id,
                            "query_type": p.query_type,
                            "language": p.language,
                            "passage_index": p.passage_index,
                            "is_selected": p.is_selected,
                            "source": p.source,
                            "text": p.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Wrote {len(passages)} passages -> {dest}")
    return dict(by_lang)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Subsample MSMARCO-XI for indexing")
    parser.add_argument("--languages", default=",".join(settings.language_list))
    parser.add_argument("--max-queries", type=int, default=8000)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--selected-ratio", type=float, default=0.7)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "subsample",
    )
    args = parser.parse_args()
    langs = [p.strip() for p in args.languages.split(",") if p.strip()]
    subsample(langs, args.max_queries, args.split, args.selected_ratio, args.out)


if __name__ == "__main__":
    main()
