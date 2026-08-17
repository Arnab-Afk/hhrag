from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.harness import Pipeline
from app.schemas import PipelineResponse, TextAskRequest
from retrieval.dense import faiss_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    settings = get_settings()
    logger.info("FAISS available: %s", faiss_available())
    logger.info("Languages: %s", settings.language_list)
    try:
        pipeline = Pipeline()
        _ = pipeline.retriever
        logger.info("Retriever ready (%s shards)", len(pipeline.retriever.shards))
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
        pipeline = Pipeline()
    yield


app = FastAPI(
    title="HH RAG Goa — Voice RAG",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pipe() -> Pipeline:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        _ = pipeline.retriever
    except FileNotFoundError as exc:
        raise HTTPException(
            503,
            str(exc),
        ) from exc
    return pipeline


@app.get("/api/health")
async def health():
    settings = get_settings()
    ready = False
    shards: list[str] = []
    if pipeline is not None:
        try:
            shards = list(pipeline.retriever.shards)
            ready = True
        except FileNotFoundError:
            ready = False
    return {
        "ok": True,
        "ready": ready,
        "shards": shards,
        "faiss": faiss_available(),
        "stt_configured": bool(settings.sarvam_api_key),
        "llm_configured": bool(settings.groq_api_key),
        "embedding_model": settings.embedding_model,
    }


@app.post("/api/ask", response_model=PipelineResponse)
async def ask_text(body: TextAskRequest) -> PipelineResponse:
    return await _pipe().run(
        query=body.query,
        language=body.language,
        mode=body.mode,
        top_k=body.top_k,
    )


@app.post("/api/ask/voice", response_model=PipelineResponse)
async def ask_voice(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    mode: str = Form(default="auto"),
) -> PipelineResponse:
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "Empty audio")
    if mode not in {"auto", "extractive", "generative"}:
        mode = "auto"
    return await _pipe().run(
        audio=audio,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "audio/webm",
        language=language,
        mode=mode,  # type: ignore[arg-type]
    )


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND / "index.html")
