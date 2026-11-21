"""Bounded async persistence queue with retries outside live audio processing."""

import asyncio
import logging
from backend.app.services.audit import AuditService

logger = logging.getLogger(__name__)


class PersistenceService:
    def __init__(self, repository, settings, memory=None, promotion=None):
        self.repository = repository
        self.settings = settings
        self.audit = AuditService()
        self.queue = asyncio.Queue(maxsize=settings.persistence_queue_size)
        self.task = None
        self.failed_jobs = 0
        self.accepting = True
        self.memory = memory
        self.promotion = promotion

    def start(self):
        self.task = asyncio.create_task(self._run(), name="discussion-persistence")

    def submit(self, turn=None, classification=None, result=None, legacy=None) -> bool:
        if not self.accepting:
            return False
        try:
            self.queue.put_nowait((turn, classification, result, legacy))
            return True
        except asyncio.QueueFull:
            self.failed_jobs += 1
            logger.error(
                "Persistence queue full; job rejected (utterance=%s)",
                getattr(turn, "utterance_id", None),
            )
            return False

    async def _write(self, job):
        turn, classification, result, legacy = job
        async with self.repository.database.session_factory() as db:
            async with db.begin():
                if legacy is not None:
                    from backend.app.repositories.transcripts import add_legacy

                    add_legacy(db, legacy)
                if turn is not None:
                    await self.repository.ensure_turn(db, turn)
                    if classification is not None:
                        await self.repository.save_classification(
                            db, turn, classification, self.audit
                        )
                    if result is not None:
                        await self.repository.save_result(db, turn, result)
        # This callback only enqueues a cache update, after the transaction commits.
        if turn is not None and self.memory is not None:
            self.memory.committed(turn)
        if result is not None and self.promotion is not None:
            self.promotion.result_committed(result["id"])

    async def _run(self):
        while True:
            job = await self.queue.get()
            try:
                for attempt in range(self.settings.persistence_retries + 1):
                    try:
                        await asyncio.wait_for(
                            self._write(job), self.settings.persistence_timeout_sec
                        )
                        break
                    except Exception:
                        if attempt == self.settings.persistence_retries:
                            self.failed_jobs += 1
                            logger.exception(
                                "Persistence exhausted retries; job lost (utterance=%s)",
                                getattr(job[0], "utterance_id", None),
                            )
                        else:
                            await asyncio.sleep(
                                self.settings.persistence_retry_sec * 2**attempt
                            )
            finally:
                self.queue.task_done()

    async def close(self):
        self.accepting = False
        if self.task is None:
            return
        try:
            await asyncio.wait_for(
                self.queue.join(), self.settings.persistence_shutdown_sec
            )
        except TimeoutError:
            logger.error(
                "Persistence shutdown timed out; %s jobs remain", self.queue.qsize()
            )
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
