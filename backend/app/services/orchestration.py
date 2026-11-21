"""Context -> classifier -> selected pipeline, outside audio capture tasks."""

import asyncio
import logging
import time
from uuid import uuid4, UUID
from backend.app.schemas.discussion import ClassifyRequest, ClassifyResponse, utcnow
from backend.app.schemas.events import LiveEvent

logger = logging.getLogger(__name__)


class OrchestrationEngine:
    def __init__(self, context, classifier, pipelines, persistence, settings):
        self.context = context
        self.classifier = classifier
        self.pipelines = pipelines
        self.persistence = persistence
        self.settings = settings
        self.session_id = uuid4()
        self.queue = asyncio.Queue(maxsize=settings.orchestration_queue_size)
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self._run(), name="discussion-orchestration")

    async def classify(self, turn):
        context, queued = self.context.receive(turn)
        context = await self.context.read(turn, context)
        classification = await self.classifier.classify(turn, context)
        queued = (
            self.persistence.submit(turn=turn, classification=classification) and queued
        )
        return (
            ClassifyResponse(
                session_id=turn.session_id,
                utterance_id=turn.utterance_id,
                classification=classification,
                persistence="queued" if queued else "unavailable",
            ),
            context,
        )

    def enqueue(self, event, publish):
        turn = ClassifyRequest(
            session_id=self.session_id,
            utterance_id=UUID(event.event_id),
            mic_id=event.mic_id,
            display_name=event.speaker,
            raw_text=event.text,
            started_at=event.timestamp,
            ended_at=utcnow(),
            stt_model=self.settings.whisper_model,
        )
        # Save on context reception even if the agent queue is overloaded.
        context, _ = self.context.receive(turn)
        try:
            self.queue.put_nowait((turn, context, event, publish))
        except asyncio.QueueFull:
            logger.error(
                "Orchestration queue full; analysis skipped for %s", turn.utterance_id
            )

    async def _run(self):
        while True:
            turn, context, event, publish = await self.queue.get()
            try:
                context = await self.context.read(turn, context)
                classification = await self.classifier.classify(turn, context)
                self.persistence.submit(turn=turn, classification=classification)
                started = time.perf_counter()
                result = {
                    "id": uuid4(),
                    "pipeline": classification.intent.value,
                    "output": {},
                    "status": "completed",
                    "latency_ms": 0.0,
                }
                try:
                    result["output"] = await asyncio.wait_for(
                        self.pipelines[classification.intent].run(turn, context),
                        timeout=self.settings.llm_timeout_sec,
                    )
                except Exception as error:
                    result.update(
                        status="failed", output={"error": type(error).__name__}
                    )
                    logger.exception(
                        "Discussion pipeline failed for %s", turn.utterance_id
                    )
                result["latency_ms"] = (time.perf_counter() - started) * 1000
                self.persistence.submit(turn=turn, result=result)
                await publish(
                    LiveEvent(
                        type=(
                            "reasoning" if result["status"] == "completed" else "error"
                        ),
                        mic_id=turn.mic_id,
                        speaker=event.speaker,
                        text=result["output"].get("text", "Analysis unavailable"),
                        source_text=turn.raw_text,
                        source_event_id=event.event_id,
                    )
                )
            except Exception:
                logger.exception(
                    "Intent classification failed for %s", turn.utterance_id
                )
                await publish(
                    LiveEvent(
                        type="error",
                        mic_id=turn.mic_id,
                        speaker=event.speaker,
                        text="Intent analysis unavailable; transcription continues.",
                        source_text=turn.raw_text,
                        source_event_id=event.event_id,
                    )
                )
            finally:
                self.queue.task_done()

    async def close(self):
        if self.task:
            try:
                await asyncio.wait_for(
                    self.queue.join(), self.settings.persistence_shutdown_sec
                )
            except TimeoutError:
                logger.error(
                    "Orchestration shutdown timed out; %s turns remain",
                    self.queue.qsize(),
                )
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
