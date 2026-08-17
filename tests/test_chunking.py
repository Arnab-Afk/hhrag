from chunking.dedup import dedup_texts, hamming, simhash64
from chunking.pipeline import build_chunks_for_passage, chunk_corpus
from chunking.script_aware import split_sentences
from chunking.types import RawPassage


def test_danda_sentence_split():
    text = "ताज महल आगरा में है। यह शाहजहाँ ने बनवाया था।"
    parts = split_sentences(text, "hi")
    assert len(parts) == 2
    assert "आगरा" in parts[0]


def test_decimal_not_split():
    text = "GST rates include 18.0 percent and 28 percent."
    parts = split_sentences(text, "en")
    assert len(parts) == 1


def test_short_passage_is_native():
    passage = RawPassage(
        query_id=1,
        query_type="DESCRIPTION",
        language="en",
        passage_index=0,
        is_selected=1,
        source="demo",
        text="UPI is a real-time payment system built by NPCI.",
    )
    chunks = build_chunks_for_passage(passage)
    assert len(chunks) == 1
    assert chunks[0].strategy == "passage_native"
    assert chunks[0].parent_text == chunks[0].text


def test_long_passage_splits():
    sentence = "The monsoon covers the subcontinent in stages. "
    text = sentence * 40
    passage = RawPassage(
        query_id=2,
        query_type="DESCRIPTION",
        language="en",
        passage_index=0,
        is_selected=0,
        source="english",
        text=text,
    )
    chunks = build_chunks_for_passage(passage)
    assert len(chunks) > 1
    assert chunks[0].parent_text == text.strip() or chunks[0].parent_id.endswith("_english")
    assert all(c.parent_id == chunks[0].parent_id for c in chunks)


def test_dedup_near_duplicates():
    texts = [
        "The Taj Mahal is in Agra, Uttar Pradesh, India.",
        "The Taj Mahal is in Agra, Uttar Pradesh, India!",
        "UPI was developed by NPCI for real-time payments.",
    ]
    keep = dedup_texts(texts, max_hamming=3)
    assert keep == [0, 2]
    assert hamming(simhash64(texts[0]), simhash64(texts[1])) <= 3


def test_chunk_corpus_drops_dupes():
    a = RawPassage(1, "LOCATION", "en", 0, 1, "demo", "The Taj Mahal is in Agra.")
    b = RawPassage(9, "LOCATION", "en", 1, 0, "demo", "The Taj Mahal is in Agra.")
    chunks = chunk_corpus([a, b])
    assert len({c.parent_id for c in chunks}) == 1
