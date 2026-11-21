"""Asynchronous worker for one microphone's streaming pipeline."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum

import numpy as np

from backend.app.core.config import Settings
from backend.app.pipelines.base import Pipeline
from backend.app.schemas.events import LiveEvent

logger = logging.getLogger(__name__)
BYTES_PER_SAMPLE = 4
MIN_FINAL_AUDIO_SECONDS = 0.35


class PipelineStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"


class PipelineWorker:
    """Preserve incoming audio while Whisper processes coalesced snapshots."""

    def __init__(
        self,
        microphone_id: int,
        pipeline: Pipeline,
        on_event: Callable[[LiveEvent], Awaitable[None]],
        settings: Settings,
    ) -> None:
        self.microphone_id = microphone_id
        self._pipeline = pipeline
        self.on_event = on_event
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=settings.audio_queue_size
        )
        self._buffer = bytearray()
        self._active = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._state_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._sample_rate = settings.audio_sample_rate
        bytes_per_second = self._sample_rate * BYTES_PER_SAMPLE
        self._window_bytes = int(bytes_per_second * settings.streaming_window_sec)
        self._step_bytes = int(bytes_per_second * settings.streaming_update_sec)
        self._silence_seconds = settings.sentence_silence_sec
        self._speech_threshold = settings.speech_rms_threshold
        self._max_buffer_bytes = int(bytes_per_second * settings.max_utterance_sec)
        self._min_final_bytes = int(bytes_per_second * MIN_FINAL_AUDIO_SECONDS)
        self._bytes_since_process = 0
        self._consecutive_silence = 0.0
        self._has_speech = False

    @property
    def status(self) -> PipelineStatus:
        return PipelineStatus.RUNNING if self._active.is_set() else PipelineStatus.IDLE

    def start(self) -> None:
        self._active.set()
        logger.info("Started pipeline for microphone %s", self.microphone_id)

    async def stop(self) -> None:
        self._active.clear()
        await self.flush()
        logger.info("Stopped pipeline for microphone %s", self.microphone_id)

    async def flush(self) -> None:
        """Finalize all audio received before a WebSocket flush command."""
        async with self._state_lock:
            chunks = self._drain_queue()
            final_audio = self._ingest_chunks(chunks)
            if self._has_speech:
                final_audio.append(bytes(self._buffer))
            self._reset_utterance()
            for audio in final_audio:
                await self._process_final_audio(audio)
            if not final_audio:
                await self._publish_events(await self._pipeline.finalize_pending())

    async def submit(self, chunk: bytes) -> None:
        if not self._active.is_set():
            return
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            logger.warning("Dropped oldest audio for microphone %s", self.microphone_id)
        self._queue.put_nowait(chunk)

    def launch(self) -> None:
        self._task = asyncio.create_task(
            self._run(), name=f"microphone-{self.microphone_id}"
        )

    async def shutdown(self) -> None:
        await self.stop()
        self._shutdown.set()
        self._active.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._pipeline.shutdown()

    def _drain_queue(self) -> list[bytes]:
        chunks: list[bytes] = []
        while not self._queue.empty():
            chunks.append(self._queue.get_nowait())
            self._queue.task_done()
        return chunks

    async def _run(self) -> None:
        await self._pipeline.start()
        while not self._shutdown.is_set():
            await self._active.wait()
            if self._shutdown.is_set():
                break
            first_chunk = await self._queue.get()
            chunks = [first_chunk]
            chunks.extend(self._drain_queue())
            try:
                async with self._state_lock:
                    await self._process_batch(chunks)
            except Exception:
                logger.exception("Pipeline error for microphone %s", self.microphone_id)
            finally:
                self._queue.task_done()

    async def _process_batch(self, chunks: list[bytes]) -> None:
        """Ingest all queued audio, then transcribe only the newest state."""
        final_audio = self._ingest_chunks(chunks)
        for audio in final_audio:
            await self._process_final_audio(audio)
        await self._maybe_publish_partial()

    def _ingest_chunks(self, chunks: list[bytes]) -> list[bytes]:
        final_audio: list[bytes] = []
        for chunk in chunks:
            self._record_chunk(chunk)
            reached_boundary = (
                self._has_speech
                and self._consecutive_silence >= self._silence_seconds
            )
            reached_limit = len(self._buffer) >= self._max_buffer_bytes
            if reached_boundary or reached_limit:
                final_audio.append(bytes(self._buffer))
                self._reset_utterance()
            elif (
                not self._has_speech
                and self._consecutive_silence >= self._silence_seconds
            ):
                self._reset_utterance()
        return final_audio

    async def _handle_chunk(self, chunk: bytes) -> None:
        """Process one chunk directly; retained for focused pipeline checks."""
        await self._process_batch([chunk])

    def _record_chunk(self, chunk: bytes) -> None:
        usable_bytes = len(chunk) - (len(chunk) % BYTES_PER_SAMPLE)
        if usable_bytes == 0:
            return
        chunk = chunk[:usable_bytes]
        samples = np.frombuffer(chunk, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        duration = samples.size / self._sample_rate

        self._buffer.extend(chunk)
        self._bytes_since_process += len(chunk)
        if rms >= self._speech_threshold:
            self._has_speech = True
            self._consecutive_silence = 0.0
        else:
            self._consecutive_silence += duration

    async def _maybe_publish_partial(self) -> None:
        if (
            self._has_speech
            and len(self._buffer) >= self._window_bytes
            and self._bytes_since_process >= self._step_bytes
        ):
            self._bytes_since_process = 0
            await self._publish_events(
                await self._pipeline.process(bytes(self._buffer))
            )

    async def _process_final_audio(self, audio: bytes) -> None:
        if len(audio) >= self._min_final_bytes:
            try:
                events = await self._pipeline.process_final(audio)
                await self._publish_events(events)
            except Exception:
                logger.exception(
                    "Final transcription update failed for microphone %s",
                    self.microphone_id,
                )
        await self._publish_events(await self._pipeline.finalize_pending())

    async def _publish_events(self, events: list[LiveEvent]) -> None:
        for event in events:
            await self.on_event(event)
            if event.type == "transcript.final":
                await self._pipeline.on_final(event)

    def _reset_utterance(self) -> None:
        self._buffer.clear()
        self._bytes_since_process = 0
        self._consecutive_silence = 0.0
        self._has_speech = False
