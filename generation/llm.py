from __future__ import annotations

import json
import logging
import re

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.schemas import Chunk, GenResult
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class GenerationError(RuntimeError):
    pass


def extractive_answer(query: str, chunks: list[Chunk]) -> GenResult:
    """No-LLM fallback: return the top parent passage as the answer."""
    if not chunks:
        return GenResult(
            answer="I don't have information on that in this dataset.",
            citation_ids=[],
            grounded=False,
            confidence=0.0,
            mode="refused",
        )
    top = chunks[0]
    return GenResult(
        answer=top.parent_text,
        citation_ids=[top.chunk_id],
        grounded=True,
        confidence=min(0.9, 0.4 + top.score * 8),
        mode="extractive",
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _result_from_payload(data: dict) -> GenResult:
    citations = data.get("citation_ids") or []
    if isinstance(citations, str):
        citations = [citations]
    return GenResult(
        answer=str(data.get("answer") or "").strip(),
        citation_ids=[str(c) for c in citations],
        grounded=bool(data.get("grounded", True)),
        confidence=float(data.get("confidence") or 0.5),
        mode="generative",
    )


async def _generate_anthropic(query: str, chunks: list[Chunk]) -> GenResult:
    settings = get_settings()
    payload_chunks = [c.model_dump() for c in chunks[:6]]
    body = {
        "model": settings.anthropic_model,
        "temperature": 0.1,
        "max_tokens": 400,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": build_user_prompt(query, payload_chunks)},
        ],
    }
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.generate_timeout_s) as client:
        response = await client.post(
            f"{settings.anthropic_base_url}/v1/messages",
            headers=headers,
            json=body,
        )
    if response.status_code >= 400:
        logger.warning("Anthropic error %s: %s", response.status_code, response.text[:400])
        raise GenerationError(f"Generation failed ({response.status_code})")

    parts = response.json().get("content") or []
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    try:
        data = _parse_json(text)
    except json.JSONDecodeError as exc:
        raise GenerationError("Model did not return JSON") from exc
    return _result_from_payload(data)


async def _generate_groq(query: str, chunks: list[Chunk]) -> GenResult:
    settings = get_settings()
    payload_chunks = [c.model_dump() for c in chunks[:6]]
    body = {
        "model": settings.groq_model,
        "temperature": 0.1,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, payload_chunks)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.generate_timeout_s) as client:
        response = await client.post(
            f"{settings.groq_base_url}/chat/completions",
            headers=headers,
            json=body,
        )
    if response.status_code >= 400:
        logger.warning("Groq error %s: %s", response.status_code, response.text[:400])
        raise GenerationError(f"Generation failed ({response.status_code})")

    content = response.json()["choices"][0]["message"]["content"]
    try:
        data = _parse_json(content)
    except json.JSONDecodeError as exc:
        raise GenerationError("Model did not return JSON") from exc
    return _result_from_payload(data)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
    retry=retry_if_exception_type((httpx.HTTPError, GenerationError)),
    reraise=True,
)
async def generate_answer(query: str, chunks: list[Chunk]) -> GenResult:
    settings = get_settings()
    if settings.anthropic_api_key:
        return await _generate_anthropic(query, chunks)
    if settings.groq_api_key:
        return await _generate_groq(query, chunks)
    return extractive_answer(query, chunks)
