"""Persistence models for recorded podcast data."""

from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
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
