"""Validated contracts between classification, orchestration and persistence."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Intent(StrEnum):
    QUESTION_QUERY = "question_query"
    FACT_CLAIM = "fact_claim"
    OPINION_DEBATE = "opinion_debate"


class ClassifyRequest(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    utterance_id: UUID = Field(default_factory=uuid4)
    mic_id: int = Field(ge=1, le=2)
    display_name: str = Field(default="Speaker", min_length=1, max_length=80)
    raw_text: str = Field(min_length=1, max_length=20000)
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime = Field(default_factory=utcnow)
    stt_model: str | None = None
    stt_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_turn(self):
        if not self.raw_text.strip():
            raise ValueError("raw_text must contain speech")
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        return self


class Classification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    model_version: str
    raw_output: dict[str, Any]
    latency_ms: float = Field(ge=0)


class ClassifyResponse(BaseModel):
    session_id: UUID
    utterance_id: UUID
    classification: Classification
    persistence: str


class UtteranceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    session_id: UUID
    participant_id: UUID
    raw_text: str
    started_at: datetime
    ended_at: datetime
    stt_model: str | None
    stt_confidence: float | None


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    utterance_id: UUID
    intent: Intent
    confidence: float
    model_version: str
    raw_output: dict[str, Any]
    latency_ms: float
    created_at: datetime
