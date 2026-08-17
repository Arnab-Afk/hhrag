from __future__ import annotations

import logging
from io import BytesIO

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.schemas import Transcript

logger = logging.getLogger(__name__)

_RETRY = dict(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=1.5),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)


class TranscriptionError(RuntimeError):
    pass


@retry(**_RETRY)
async def transcribe_audio(
    audio: bytes,
    filename: str = "audio.webm",
    content_type: str = "audio/webm",
    language_code: str = "unknown",
) -> Transcript:
    settings = get_settings()
    if not settings.sarvam_api_key:
        raise TranscriptionError(
            "SARVAM_API_KEY is not set. Add it to .env (see .env.example)."
        )
    if not audio:
        raise TranscriptionError("Empty audio payload.")

    headers = {"api-subscription-key": settings.sarvam_api_key}
    data = {
        "model": settings.sarvam_model,
        "mode": settings.sarvam_mode,
        "language_code": language_code,
    }
    files = {"file": (filename, BytesIO(audio), content_type)}

    async with httpx.AsyncClient(timeout=settings.stt_timeout_s) as client:
        response = await client.post(
            settings.sarvam_stt_url,
            headers=headers,
            data=data,
            files=files,
        )
    if response.status_code >= 400:
        logger.warning("Sarvam STT error %s: %s", response.status_code, response.text[:400])
        raise TranscriptionError(f"Sarvam STT failed ({response.status_code})")

    payload = response.json()
    text = (payload.get("transcript") or "").strip()
    if not text:
        raise TranscriptionError("Empty transcript.")
    return Transcript(
        text=text,
        language_code=payload.get("language_code"),
        language_probability=payload.get("language_probability"),
        request_id=payload.get("request_id"),
    )
