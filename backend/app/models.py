"""Persistence models for recorded podcast data."""

from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    Uuid,
    Float,
    Enum,
    UniqueConstraint,
    CheckConstraint,
    Numeric,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for persistence models."""


class PodcastSession(Base):
    """A podcast recording session."""

    __tablename__ = "podcast_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TranscriptSegment(Base):
    """A finalized spoken segment."""

    __tablename__ = "transcript_segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    mic_id: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(80))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReasoningEntry(Base):
    """A Groq fact-check response linked to a microphone."""

    __tablename__ = "reasoning_entries"
    id: Mapped[int] = mapped_column(primary_key=True)
    mic_id: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(80))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class Participant(Base):
    __tablename__ = "participants"
    __table_args__ = (UniqueConstraint("session_id", "mic_id"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id"), index=True)
    mic_id: Mapped[int] = mapped_column(Integer)
    display_name: Mapped[str] = mapped_column(String(80))


class Utterance(Base):
    __tablename__ = "utterances"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id"), index=True)
    participant_id: Mapped[UUID] = mapped_column(
        ForeignKey("participants.id"), index=True
    )
    raw_text: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stt_model: Mapped[str | None] = mapped_column(String(120))
    stt_confidence: Mapped[float | None] = mapped_column(Float)


class IntentClassification(Base):
    __tablename__ = "intent_classifications"
    __table_args__ = (CheckConstraint("confidence >= 0 AND confidence <= 1"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    utterance_id: Mapped[UUID] = mapped_column(
        ForeignKey("utterances.id"), unique=True, index=True
    )
    intent: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(160))
    raw_output: Mapped[dict] = mapped_column(JSONB)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class PipelineResult(Base):
    __tablename__ = "pipeline_results"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    utterance_id: Mapped[UUID] = mapped_column(ForeignKey("utterances.id"), index=True)
    pipeline: Mapped[str] = mapped_column(
        Enum("question_query", "fact_claim", "opinion_debate", name="pipeline_kind")
    )
    output: Mapped[dict] = mapped_column(JSONB)
    latency_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(80))
    payload: Mapped[dict] = mapped_column(JSONB)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class LongTermEvidence(Base):
    __tablename__ = "ltm_evidence"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id"), index=True)
    utterance_id: Mapped[UUID] = mapped_column(ForeignKey("utterances.id"), index=True)
    source_result_id: Mapped[UUID] = mapped_column(
        ForeignKey("pipeline_results.id"), unique=True
    )
    claim: Mapped[str] = mapped_column(Text)
    fact_status: Mapped[str] = mapped_column(
        Enum("true", "false", "uncertain", name="fact_status"), index=True
    )
    sources: Mapped[list] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class LongTermDecision(Base):
    __tablename__ = "ltm_decisions"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id"), index=True)
    utterance_id: Mapped[UUID] = mapped_column(ForeignKey("utterances.id"), index=True)
    source_result_id: Mapped[UUID] = mapped_column(
        ForeignKey("pipeline_results.id"), unique=True
    )
    agent: Mapped[str] = mapped_column(String(80))
    decision: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class LongTermEthics(Base):
    __tablename__ = "ltm_ethics"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    utterance_id: Mapped[UUID] = mapped_column(
        ForeignKey("utterances.id"), unique=True, index=True
    )
    ethical_score: Mapped[float | None] = mapped_column(Numeric)
    dimensions: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
