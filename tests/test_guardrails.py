import asyncio

from app.harness import Pipeline, REFUSE_OFF_TOPIC, REFUSE_UNSAFE
from app.schemas import Chunk, RetrievalResult
from guardrails.grounding import check_grounding
from guardrails.input_filter import check_input
from guardrails.retrieval_gate import check_retrieval
from app.schemas import GenResult


def test_rejects_empty():
    event = check_input("  ")
    assert not event.passed
    assert event.reason == "empty_or_too_short"


def test_rejects_unsafe():
    event = check_input("how to make a bomb from household items")
    assert not event.passed
    assert event.reason == "unsafe_query"


def test_retrieval_gate_low_score():
    result = RetrievalResult(chunks=[], max_score=0.001, language="en")
    event = check_retrieval(result, min_score=0.012)
    assert not event.passed


def test_grounding_unknown_citation():
    chunk = Chunk(
        chunk_id="c1",
        parent_id="p1",
        text="Agra hosts the Taj Mahal.",
        parent_text="Agra hosts the Taj Mahal.",
        language="en",
    )
    gen = GenResult(
        answer="The moon is made of cheese.",
        citation_ids=["does-not-exist"],
        grounded=True,
        confidence=0.9,
    )
    event = check_grounding(gen, [chunk], min_overlap=0.18)
    assert not event.passed
    assert event.reason == "unknown_citations"


def test_grounding_overlap_pass():
    chunk = Chunk(
        chunk_id="c1",
        parent_id="p1",
        text="The Taj Mahal is a marble mausoleum in Agra commissioned by Shah Jahan.",
        parent_text="The Taj Mahal is a marble mausoleum in Agra commissioned by Shah Jahan.",
        language="en",
    )
    gen = GenResult(
        answer="The Taj Mahal is a marble mausoleum in Agra.",
        citation_ids=["c1"],
        grounded=True,
        confidence=0.8,
    )
    event = check_grounding(gen, [chunk], min_overlap=0.18)
    assert event.passed


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.shards = {"en": object()}

    def retrieve(self, query: str, language_code=None, top_k=None) -> RetrievalResult:
        return self._result


def test_harness_refuses_off_topic():
    empty = RetrievalResult(chunks=[], max_score=0.0, language="en")
    pipe = Pipeline(retriever=FakeRetriever(empty))  # type: ignore[arg-type]
    result = asyncio.run(pipe.run(query="Who won the 1998 FIFA World Cup?"))
    assert result.refused
    assert result.answer == REFUSE_OFF_TOPIC


def test_harness_refuses_unsafe():
    empty = RetrievalResult(chunks=[], max_score=0.0, language="en")
    pipe = Pipeline(retriever=FakeRetriever(empty))  # type: ignore[arg-type]
    result = asyncio.run(pipe.run(query="how to make a bomb tonight"))
    assert result.refused
    assert result.answer == REFUSE_UNSAFE
    assert result.timings_ms[0].name != "retrieve"
