from retrieval.fusion import rrf_fuse
from retrieval.language import shard_for_language_code
from retrieval.query_type import guess_query_type


def test_rrf_prefers_agreement():
    dense = [10, 11, 12]
    sparse = [12, 10, 99]
    fused = rrf_fuse(dense, sparse, k=60)
    # id 10 and 12 appear in both lists and should beat 99
    assert fused[10] > fused[99]
    assert fused[12] > fused[11]


def test_language_routing():
    assert shard_for_language_code("hi-IN") == "hi"
    assert shard_for_language_code("en-IN") == "en"
    assert shard_for_language_code("od-IN") == "or"
    assert shard_for_language_code("unknown") is None


def test_query_type_numeric():
    assert guess_query_type("How many overs in T20?") == "NUMERIC"
    assert guess_query_type("कितना GST है?") == "NUMERIC"
    assert guess_query_type("Who drafted the constitution?") == "PERSON"


def test_msmarco_parquet_paths():
    from ingest.hf_source import parquet_path

    assert parquet_path("hi", "train") == "train/hintrain.parquet"
    assert parquet_path("hi", "validation") == "validation/hinval.parquet"
    assert parquet_path("gu", "train") == "train/gujtrain.parquet"
    assert parquet_path("or", "validation") == "validation/orival.parquet"
    try:
        parquet_path("te", "train")
        raise AssertionError("expected missing Telugu train file")
    except FileNotFoundError:
        pass
    from ingest.hf_source import TRAIN_LANGUAGES

    assert "hi" in TRAIN_LANGUAGES
    assert "te" not in TRAIN_LANGUAGES


def test_rescore_bm25_ranks_lexical_match_first():
    from retrieval.sparse import rescore_bm25

    docs = [
        "the weather in paris is mild",
        "taj mahal was built by shah jahan in agra",
        "unrelated cooking recipe",
    ]
    order = rescore_bm25("who built the taj mahal", docs)
    assert order[0] == 1


def test_chunk_offsets_roundtrip(tmp_path):
    from retrieval.chunk_store import build_line_offsets, ensure_offsets, read_rows

    jsonl = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "a", "text": "first"},
        {"chunk_id": "b", "text": ""},
        {"chunk_id": "c", "text": "third"},
    ]
    import orjson

    with jsonl.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row) + b"\n")
    offsets = build_line_offsets(jsonl)
    assert len(offsets) == 2
    loaded = ensure_offsets(tmp_path)
    assert len(loaded) == 2
    fetched = read_rows(jsonl, loaded, [1, 0])
    assert fetched[1]["text"] == "third"
    assert fetched[0]["text"] == "first"
