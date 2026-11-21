"""Faster-Whisper adapter for revisable live transcripts."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

import live_whisper_diarize as live_whisper
from backend.app.core.config import Settings
from backend.app.pipelines.base import Pipeline
from backend.app.schemas.events import LiveEvent

logger = logging.getLogger(__name__)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


class WhisperPipeline(Pipeline):
    """Transcribe the active utterance and analyze it once after finalization."""

    def __init__(self, microphone_id: int, settings: Settings) -> None:
        self._microphone_id = microphone_id
        self._settings = settings
        self._model: Any | None = None
        self._last_partial = ""
        self._utterance_id = uuid4().hex
        self._utterance_timestamp = None
        self._publish_event: Callable[[LiveEvent], Awaitable[None]] | None = None
        self._analysis_tasks: set[asyncio.Task[None]] = set()

    def set_event_publisher(
        self, callback: Callable[[LiveEvent], Awaitable[None]]
    ) -> None:
        self._publish_event = callback

    async def start(self) -> None:
        live_whisper.configure_llm(
            self._settings.groq_api_key, self._settings.groq_model
        )
        self._model = await asyncio.to_thread(self._create_legacy_model)

    def _create_legacy_model(self) -> Any:
        device = (
            None
            if self._settings.whisper_device == "auto"
            else self._settings.whisper_device
        )
        return live_whisper.create_whisper_model(
            model_size=self._settings.whisper_model,
            device=device,
            compute_type=self._settings.whisper_compute_type,
        )

    async def process(self, chunk: Any) -> list[LiveEvent]:
        """Return a replacement for the currently visible partial transcript."""
        return await self._process_audio(chunk, beam_size=1)

    async def process_final(self, chunk: Any) -> list[LiveEvent]:
        """Decode a completed utterance more accurately than a live partial."""
        return await self._process_audio(
            chunk, beam_size=self._settings.whisper_final_beam_size
        )

    async def _process_audio(self, chunk: Any, beam_size: int) -> list[LiveEvent]:
        if self._utterance_timestamp is None:
            self._utterance_timestamp = datetime.now(timezone.utc)
        audio = np.frombuffer(chunk, dtype=np.float32)
        if audio.size == 0:
            return []
        segments = await asyncio.to_thread(self._transcribe, audio, beam_size)
        text = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        if self._settings.whisper_language == "hinglish" and text:
            text = live_whisper.devanagari_to_hinglish(text)
        if not text or text == self._last_partial:
            return []
        self._last_partial = text
        return [self._event("transcript.partial", text).model_copy(update={
            "event_id": self._utterance_id,
            "timestamp": self._utterance_timestamp,
        })]

    def _transcribe(self, audio: np.ndarray, beam_size: int) -> list[Any]:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded")
        segments, _ = live_whisper.transcribe_audio(
            model=self._model,
            audio=audio,
            language_mode=self._settings.whisper_language,
            task=self._settings.whisper_task,
            beam_size=beam_size,
            initial_prompt=self._settings.whisper_initial_prompt,
        )
        return segments

    async def finalize_pending(self) -> list[LiveEvent]:
        """Convert the current partial into exactly one finalized sentence."""
        if not self._last_partial:
            return []
        text, self._last_partial = self._last_partial, ""
        sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text)]
        events = [
            self._event("transcript.final", sentence)
            for sentence in sentences
            if sentence
        ]
        for index, event in enumerate(events):
            event.timestamp = self._utterance_timestamp or event.timestamp
            if index == 0:
                event.event_id = self._utterance_id
        self._utterance_id = uuid4().hex
        self._utterance_timestamp = None
        return events

    async def on_final(self, event: LiveEvent) -> None:
        """Start analysis only after the finalized transcript was published."""
        if not self._settings.groq_api_key:
            return
        task = asyncio.create_task(
            self._analyze(event),
            name=f"reasoning-mic-{self._microphone_id}",
        )
        self._analysis_tasks.add(task)
        task.add_done_callback(self._analysis_tasks.discard)

    async def _analyze(self, source: LiveEvent) -> None:
        statement = source.text
        try:
            reasoning = await asyncio.wait_for(
                asyncio.to_thread(live_whisper.fact_check_statement, statement),
                timeout=self._settings.llm_timeout_sec,
            )
            await self._publish(
                self._event("reasoning", reasoning, source_text=statement).model_copy(
                    update={"source_event_id": source.event_id}
                )
            )
        except TimeoutError:
            await self._publish(
                self._event(
                    "error",
                    "Reasoning timed out; live transcription is still running.",
                    source_text=statement,
                )
            )
        except Exception as error:
            logger.exception(
                "Reasoning failed for microphone %s", self._microphone_id
            )
            await self._publish(
                self._event(
                    "error",
                    f"Reasoning unavailable: {error}",
                    source_text=statement,
                )
            )

    async def _publish(self, event: LiveEvent) -> None:
        if self._publish_event is not None:
            await self._publish_event(event)

    def _event(
        self, event_type: str, text: str, source_text: str | None = None
    ) -> LiveEvent:
        return LiveEvent(
            type=event_type,
            mic_id=self._microphone_id,
            speaker=f"Mic {self._microphone_id}",
            text=text,
            source_text=source_text,
        )

    async def shutdown(self) -> None:
        for task in list(self._analysis_tasks):
            task.cancel()
        if self._analysis_tasks:
            await asyncio.gather(*self._analysis_tasks, return_exceptions=True)
