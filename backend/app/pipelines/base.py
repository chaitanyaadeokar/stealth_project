"""Pipeline abstractions and the initial no-op implementation."""

from abc import ABC, abstractmethod
from typing import Any
from backend.app.schemas.events import LiveEvent


class Pipeline(ABC):
    """Contract for processing audio chunks from one microphone."""

    async def start(self) -> None:
        """Prepare resources before the worker accepts audio."""

        return None

    @abstractmethod
    async def process(self, chunk: Any) -> list[LiveEvent]:
        """Process one audio chunk for this pipeline instance."""

    async def process_final(self, chunk: Any) -> list[LiveEvent]:
        """Process a completed utterance with final-decoding settings."""

        return await self.process(chunk)

    async def finalize_pending(self) -> list[LiveEvent]:
        """Finalize the current utterance, if one exists."""

        return []

    async def on_final(self, event: LiveEvent) -> None:
        """Start any work that depends on a finalized transcript."""

        return None

    async def shutdown(self) -> None:
        """Release pipeline resources and background tasks."""

        return None


class StubPipeline(Pipeline):
    """Placeholder pipeline with clear extension points for future audio work."""

    async def process(self, chunk: Any) -> list[LiveEvent]:
        """Accept a chunk without processing it until audio stages are defined."""
        # TODO: Add isolated capture, transcription, and persistence stages here.
        _ = chunk
        return []
