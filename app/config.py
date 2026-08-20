from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = ""
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "transcribe"

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_base_url: str = "https://api.anthropic.com"

    index_dir: Path = ROOT / "indexes"
    data_dir: Path = ROOT / "data"
    embedding_model: str = "intfloat/multilingual-e5-small"
    languages: str = "hi,en"

    @field_validator("index_dir", "data_dir", mode="after")
    @classmethod
    def _resolve_under_root(cls, value: Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        return path

    retrieve_top_k: int = 20
    rrf_k: int = 60
    min_retrieval_score: float = 0.012
    enable_rerank: bool = False
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    grounding_min_overlap: float = 0.18
    max_audio_seconds: int = 30

    host: str = "0.0.0.0"
    port: int = 7860

    stt_timeout_s: float = 20.0
    generate_timeout_s: float = 25.0
    retrieve_timeout_s: float = 0.2

    window_tokens: int = 128
    overlap_tokens: int = 24
    long_passage_tokens: int = 180
    semantic_sim_threshold: float = 0.42
    simhash_hamming: int = 3

    @property
    def language_list(self) -> list[str]:
        return [part.strip() for part in self.languages.split(",") if part.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.anthropic_api_key or self.groq_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
