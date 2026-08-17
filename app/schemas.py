from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    parent_text: str
    language: str
    query_id: int | None = None
    query_type: str | None = None
    is_selected: int = 0
    source: Literal["translated", "english", "demo"] = "translated"
    strategy: str = "passage_native"
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None


class Transcript(BaseModel):
    text: str
    language_code: str | None = None
    language_probability: float | None = None
    request_id: str | None = None


class RetrievalResult(BaseModel):
    chunks: list[Chunk] = Field(default_factory=list)
    max_score: float = 0.0
    language: str
    shards_searched: list[str] = Field(default_factory=list)
    query_type_guess: str | None = None


class GenResult(BaseModel):
    answer: str
    citation_ids: list[str] = Field(default_factory=list)
    grounded: bool = False
    confidence: float = 0.0
    mode: Literal["generative", "extractive", "refused"] = "generative"


class GuardrailEvent(BaseModel):
    name: str
    passed: bool
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class StageTiming(BaseModel):
    name: str
    ms: float


class PipelineResponse(BaseModel):
    ok: bool
    refused: bool = False
    refuse_reason: str | None = None
    query: str = ""
    detected_language: str | None = None
    answer: str = ""
    citations: list[Chunk] = Field(default_factory=list)
    grounded: bool = False
    confidence: float = 0.0
    generation_mode: str = "extractive"
    guardrails: list[GuardrailEvent] = Field(default_factory=list)
    timings_ms: list[StageTiming] = Field(default_factory=list)
    retrieval_ms: float = 0.0
    end_to_end_ms: float = 0.0
    retrieval_under_200ms: bool = False


class TextAskRequest(BaseModel):
    query: str
    language: str | None = None
    mode: Literal["auto", "extractive", "generative"] = "auto"
    top_k: int | None = None
