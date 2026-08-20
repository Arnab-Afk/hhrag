"""Typed pipeline harness: retries, timeouts, structured I/O, explicit refusals."""

from __future__ import annotations

import logging
from typing import Literal

from app.config import get_settings
from app.schemas import (
    Chunk,
    GuardrailEvent,
    PipelineResponse,
    RetrievalResult,
    StageTiming,
    Transcript,
)
from app.timing import Timer
from generation.llm import GenerationError, extractive_answer, generate_answer
from guardrails.grounding import check_grounding
from guardrails.input_filter import check_input
from guardrails.retrieval_gate import check_retrieval
from retrieval.hybrid import HybridRetriever
from stt.sarvam import TranscriptionError, transcribe_audio

logger = logging.getLogger(__name__)

REFUSE_OFF_TOPIC = "I don't have information on that in this dataset."
REFUSE_UNSAFE = "I can't help with that request."
REFUSE_UNGROUNDED = "I retrieved some passages, but I can't ground an answer in them."
REFUSE_STT = "I couldn't hear that clearly. Please try again."


def _response(
    *,
    timer: Timer,
    ok: bool,
    refused: bool = False,
    refuse_reason: str | None = None,
    query: str = "",
    detected_language: str | None = None,
    answer: str = "",
    citations: list[Chunk] | None = None,
    grounded: bool = False,
    confidence: float = 0.0,
    generation_mode: str = "refused",
    guardrails: list[GuardrailEvent] | None = None,
) -> PipelineResponse:
    retrieval_ms = timer.get("retrieve")
    return PipelineResponse(
        ok=ok,
        refused=refused,
        refuse_reason=refuse_reason,
        query=query,
        detected_language=detected_language,
        answer=answer,
        citations=citations or [],
        grounded=grounded,
        confidence=confidence,
        generation_mode=generation_mode,
        guardrails=guardrails or [],
        timings_ms=[StageTiming(name=n, ms=round(ms, 3)) for n, ms in timer.stages],
        retrieval_ms=round(retrieval_ms, 3),
        end_to_end_ms=round(timer.total_ms, 3),
        retrieval_under_200ms=retrieval_ms < 200.0 if retrieval_ms else False,
    )


class Pipeline:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever.load()
        return self._retriever

    async def run(
        self,
        *,
        query: str | None = None,
        audio: bytes | None = None,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str | None = None,
        mode: Literal["auto", "extractive", "generative"] = "auto",
        top_k: int | None = None,
    ) -> PipelineResponse:
        timer = Timer()
        settings = get_settings()
        guardrails: list[GuardrailEvent] = []
        detected: str | None = language
        text = (query or "").strip()

        if audio and not text:
            try:
                with timer.stage("stt"):
                    transcript: Transcript = await transcribe_audio(
                        audio,
                        filename=filename,
                        content_type=content_type,
                    )
                text = transcript.text
                detected = transcript.language_code or detected
            except TranscriptionError as exc:
                logger.info("STT failed: %s", exc)
                return _response(
                    timer=timer,
                    ok=False,
                    refused=True,
                    refuse_reason="stt_failed",
                    answer=REFUSE_STT,
                    guardrails=[
                        GuardrailEvent(name="stt", passed=False, reason=str(exc))
                    ],
                )

        with timer.stage("input_guardrail"):
            incoming = check_input(text)
        guardrails.append(incoming)
        if not incoming.passed:
            answer = REFUSE_UNSAFE if incoming.reason == "unsafe_query" else REFUSE_STT
            return _response(
                timer=timer,
                ok=True,
                refused=True,
                refuse_reason=incoming.reason,
                query=text,
                detected_language=detected,
                answer=answer,
                guardrails=guardrails,
            )

        with timer.stage("retrieve"):
            retrieval: RetrievalResult = self.retriever.retrieve(
                text, language_code=detected, top_k=top_k
            )

        with timer.stage("retrieval_guardrail"):
            gate = check_retrieval(retrieval)
        guardrails.append(gate)
        if not gate.passed:
            return _response(
                timer=timer,
                ok=True,
                refused=True,
                refuse_reason=gate.reason,
                query=text,
                detected_language=detected or retrieval.language,
                answer=REFUSE_OFF_TOPIC,
                guardrails=guardrails,
                generation_mode="refused",
            )

        use_llm = mode == "generative" or (
            mode == "auto" and settings.llm_configured
        )
        try:
            with timer.stage("generate"):
                if use_llm:
                    generation = await generate_answer(text, retrieval.chunks)
                else:
                    generation = extractive_answer(text, retrieval.chunks)
        except GenerationError as exc:
            logger.info("Generation failed, falling back to extractive: %s", exc)
            with timer.stage("generate_fallback"):
                generation = extractive_answer(text, retrieval.chunks)

        with timer.stage("grounding_guardrail"):
            grounding = check_grounding(generation, retrieval.chunks)
        guardrails.append(grounding)

        if not grounding.passed or not generation.grounded:
            return _response(
                timer=timer,
                ok=True,
                refused=True,
                refuse_reason=grounding.reason,
                query=text,
                detected_language=detected or retrieval.language,
                answer=REFUSE_UNGROUNDED,
                citations=retrieval.chunks[:3],
                grounded=False,
                confidence=0.0,
                generation_mode="refused",
                guardrails=guardrails,
            )

        cited = [
            c
            for c in retrieval.chunks
            if c.chunk_id in generation.citation_ids or c.parent_id in generation.citation_ids
        ] or retrieval.chunks[:3]

        return _response(
            timer=timer,
            ok=True,
            query=text,
            detected_language=detected or retrieval.language,
            answer=generation.answer,
            citations=cited,
            grounded=True,
            confidence=generation.confidence,
            generation_mode=generation.mode,
            guardrails=guardrails,
        )
