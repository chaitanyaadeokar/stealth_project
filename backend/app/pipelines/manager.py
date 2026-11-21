"""Coordinate microphone workers and distribute their live events."""

import asyncio
import logging
from collections import defaultdict

from backend.app.core.config import Settings
from backend.app.pipelines.whisper_pipeline import WhisperPipeline
from backend.app.pipelines.worker import PipelineStatus, PipelineWorker
from backend.app.schemas.events import LiveEvent
from backend.app.services.transcripts import TranscriptService

logger = logging.getLogger(__name__)

MICROPHONE_IDS = (1, 2)
COMBINED_STREAM_ID = 0
EVENT_HISTORY_LIMIT = 200
SUBSCRIBER_QUEUE_SIZE = 100


class PipelineManager:
    """Own isolated workers and deliver their events to live clients."""

    def __init__(
        self, settings: Settings, transcript_service: TranscriptService
    ) -> None:
        self._transcript_service = transcript_service
        self._event_history: list[LiveEvent] = []
        self._subscribers: dict[int, set[asyncio.Queue[LiveEvent]]] = defaultdict(set)
        self._workers = self._create_workers(settings)

    def _create_workers(self, settings: Settings) -> dict[int, PipelineWorker]:
        """Create one fresh pipeline and worker for each microphone."""
        workers = {}
        for microphone_id in MICROPHONE_IDS:
            pipeline = WhisperPipeline(microphone_id, settings)
            pipeline.set_event_publisher(self.publish)
            workers[microphone_id] = PipelineWorker(
                microphone_id=microphone_id,
                pipeline=pipeline,
                on_event=self.publish,
                settings=settings,
            )
        return workers

    def start_workers(self) -> None:
        """Launch the long-running task for every worker."""
        for worker in self._workers.values():
            worker.launch()

    async def shutdown_workers(self) -> None:
        """Stop every worker before application shutdown."""
        for worker in self._workers.values():
            await worker.shutdown()

    async def publish(self, event: LiveEvent) -> None:
        """Persist an event where appropriate and fan it out to clients."""
        self._remember(event)
        await self._persist_event(event)
        for queue in self._queues_for(event.mic_id):
            self._offer_event(queue, event)

    async def _persist_event(self, event: LiveEvent) -> None:
        """Persist without allowing database errors to interrupt live delivery."""
        try:
            await self._transcript_service.save(event)
        except Exception:
            logger.exception("Could not persist event from microphone %s", event.mic_id)

    def _remember(self, event: LiveEvent) -> None:
        self._event_history.append(event)
        del self._event_history[:-EVENT_HISTORY_LIMIT]

    def _queues_for(self, microphone_id: int) -> list[asyncio.Queue[LiveEvent]]:
        """Return subscribers for the mic plus subscribers of the combined feed."""
        return [
            *self._subscribers[microphone_id],
            *self._subscribers[COMBINED_STREAM_ID],
        ]

    @staticmethod
    def _offer_event(queue: asyncio.Queue[LiveEvent], event: LiveEvent) -> None:
        if queue.full():
            logger.warning("Dropping live event because a client is too slow")
            return
        queue.put_nowait(event)

    def subscribe(self, stream_id: int) -> asyncio.Queue[LiveEvent]:
        """Register one client for a microphone or the combined stream (zero)."""
        queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers[stream_id].add(queue)
        return queue

    def unsubscribe(self, stream_id: int, queue: asyncio.Queue[LiveEvent]) -> None:
        """Remove a disconnected client from its stream."""
        self._subscribers[stream_id].discard(queue)

    def start(self, microphone_id: int) -> PipelineStatus:
        """Start one microphone pipeline."""
        worker = self._worker_for(microphone_id)
        worker.start()
        return worker.status

    async def stop(self, microphone_id: int) -> PipelineStatus:
        """Pause one microphone pipeline."""
        worker = self._worker_for(microphone_id)
        await worker.stop()
        return worker.status

    def status(self, microphone_id: int) -> PipelineStatus:
        """Return one microphone pipeline's current status."""
        return self._worker_for(microphone_id).status

    async def submit_audio(self, microphone_id: int, audio: bytes) -> None:
        """Pass a PCM chunk to the worker that owns its microphone."""
        await self._worker_for(microphone_id).submit(audio)

    async def flush_audio(self, microphone_id: int) -> None:
        """Finalize audio received before the client's flush command."""
        await self._worker_for(microphone_id).flush()

    def _worker_for(self, microphone_id: int) -> PipelineWorker:
        try:
            return self._workers[microphone_id]
        except KeyError as error:
            raise ValueError("microphone_id must be 1 or 2") from error
