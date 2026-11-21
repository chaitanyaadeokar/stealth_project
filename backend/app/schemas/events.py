"""Events emitted to live clients."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, Field


class LiveEvent(BaseModel):
    """One transcript, reasoning, status, or error event."""

    type: Literal[
        "transcript.partial", "transcript.final", "reasoning", "status", "error"
    ]
    mic_id: int
    text: str
    speaker: str = "Mic"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    start_seconds: float | None = None
    end_seconds: float | None = None
    source_text: str | None = None
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    source_event_id: str | None = None
