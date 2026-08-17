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
