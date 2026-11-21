"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from the project's environment file."""

    database_url: str = "postgresql+asyncpg://postgres:mysecretpassword@localhost:5432/postgres"
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
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return the shared immutable application settings."""
    return Settings()
