"""Persistence service for live pipeline outputs."""

from backend.app.core.database import Database
from backend.app.models import ReasoningEntry, TranscriptSegment
from backend.app.schemas.events import LiveEvent


class TranscriptService:
    """Write finalized events through the reusable database session factory."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def save(self, event: LiveEvent) -> None:
        """Persist transcript finals and reasoning, leaving other events transient."""
        if event.type not in {"transcript.final", "reasoning"}:
            return
        row = self._record_for(event)
        async with self._database.session_factory() as session:
            session.add(row)
            await session.commit()

    @staticmethod
    def _record_for(event: LiveEvent) -> TranscriptSegment | ReasoningEntry:
        """Create the database record appropriate for one live event."""

        fields = {"mic_id": event.mic_id, "speaker": event.speaker, "text": event.text}
        if event.type == "transcript.final":
            return TranscriptSegment(**fields)
        return ReasoningEntry(**fields)
