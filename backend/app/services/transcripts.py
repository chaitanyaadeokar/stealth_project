"""Persistence service for live pipeline outputs."""

from backend.app.schemas.events import LiveEvent


class TranscriptService:
    """Write finalized events through the reusable database session factory."""

    def __init__(self, persistence) -> None:
        self._persistence = persistence

    async def save(self, event: LiveEvent) -> None:
        """Persist transcript finals and reasoning, leaving other events transient."""
        if event.type not in {"transcript.final", "reasoning"}:
            return
        self._persistence.submit(legacy=event)
