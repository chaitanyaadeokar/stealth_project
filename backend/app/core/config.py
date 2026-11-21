"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from the project's environment file."""

    database_url: str = (
        "postgresql+asyncpg://postgres:mysecretpassword@localhost:5432/postgres"
    )
    log_level: str = "INFO"
    whisper_model: str = "base.en"
    whisper_language: str = "en"
    whisper_task: str = "transcribe"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_final_beam_size: int = Field(default=5, ge=1, le=10)
    whisper_initial_prompt: str | None = None
    hf_token: str | None = Field(default=None, repr=False)
    modernbert_model_name: str = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
    groq_api_key: str | None = Field(default=None, repr=False)
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_sec: float = Field(default=30.0, gt=0)
    conversation_db: str = "data/sessions.sqlite3"
    vague_log_file: str = "vague.log"
    audio_sample_rate: int = Field(default=16000, gt=0)
    audio_queue_size: int = Field(default=400, gt=0)
    streaming_window_sec: float = Field(default=1.2, gt=0)
    streaming_update_sec: float = Field(default=0.4, gt=0)
    sentence_silence_sec: float = Field(default=2.0, gt=0)
    speech_rms_threshold: float = Field(default=0.006, ge=0)
    max_utterance_sec: float = Field(default=30.0, gt=0)
    intent_model: str | None = None
    context_window_turns: int = Field(default=20, ge=1, le=100)
    context_max_sessions: int = Field(default=100, ge=1)
    orchestration_queue_size: int = Field(default=100, ge=1)
    persistence_queue_size: int = Field(default=500, ge=1)
    persistence_retries: int = Field(default=3, ge=0, le=10)
    persistence_retry_sec: float = Field(default=0.25, gt=0)
    persistence_timeout_sec: float = Field(default=5.0, gt=0)
    persistence_shutdown_sec: float = Field(default=10.0, gt=0)
    memcached_host: str = "127.0.0.1"
    memcached_port: int = Field(default=11211, ge=1, le=65535)
    session_ttl_seconds: int = Field(default=3600, ge=1, le=2592000)
    memory_timeout_sec: float = Field(default=0.15, gt=0)
    memory_db_timeout_sec: float = Field(default=0.5, gt=0)
    memory_max_bytes: int = Field(default=900000, ge=1024, le=950000)
    memory_cas_retries: int = Field(default=8, ge=1, le=50)
    memory_queue_size: int = Field(default=500, ge=1)
    ltm_promote_on_result: bool = True
    ltm_min_confidence: float = Field(default=0.75, ge=0, le=1)
    ltm_min_response_chars: int = Field(default=80, ge=1)
    ltm_queue_size: int = Field(default=500, ge=1)
    ltm_job_timeout_sec: float = Field(default=30.0, gt=0)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return the shared immutable application settings."""
    return Settings()
