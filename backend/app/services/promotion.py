"""Idempotent promotion of committed results into long-term PostgreSQL memory."""

import asyncio
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def evidence_fields(output, raw_text):
    """Do not infer verified facts or citations from unstructured LLM prose."""
    evidence = output.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    claim = evidence.get("claim") or raw_text
    if not isinstance(claim, str):
        claim = raw_text
    sources = evidence.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    valid_sources = []
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            continue
        try:
            parsed = urlparse(source["url"])
            if parsed.scheme in ("http", "https") and parsed.netloc:
                valid_sources.append(source)
        except ValueError:
            continue
    sources = valid_sources
    status = evidence.get("fact_status", "uncertain")
    # Only a structured verification adapter can assert supported true/false.
    if status not in ("true", "false", "uncertain") or not any(
        source.get("verified") is True for source in sources
    ):
        status = "uncertain"
    return claim, status, sources


class MemoryPromotionService:
    def __init__(self, repository, settings):
        self.repository = repository
        self.settings = settings
        self.queue = asyncio.Queue(maxsize=settings.ltm_queue_size)
        self.task = None
        self.failed_jobs = 0

    def start(self):
        self.task = asyncio.create_task(self._run(), name="long-term-promotion")

    def _enqueue(self, kind, identifier):
        try:
            self.queue.put_nowait((kind, identifier))
            return True
        except asyncio.QueueFull:
            self.failed_jobs += 1
            logger.error(
                "Long-term promotion queue full; replay session promotion to recover"
            )
            return False

    def result_committed(self, result_id):
        return self._enqueue("result", result_id)

    def promote_session(self, session_id, end=False):
        return self._enqueue("end" if end else "session", session_id)

    async def records(self, session_id, limit=100):
        groups = await self.repository.by_session(session_id, limit)
        records = []
        for kind, rows in zip(("evidence", "decision", "ethics"), groups):
            for row in rows:
                record = {
                    column.name: getattr(row, column.name)
                    for column in row.__table__.columns
                }
                record["kind"] = kind
                records.append(record)
        return records

    async def promote_result(self, result_id, force=False):
        async with self.repository.database.session_factory() as db:
            async with db.begin():
                source = await self.repository.source(db, result_id)
                if source is None:
                    return
                result, turn, classification, ended_at = source
                if (
                    not force
                    and not self.settings.ltm_promote_on_result
                    and ended_at is None
                ):
                    return
                if result.status != "completed":
                    return
                output = result.output
                text = output.get("text", "")
                explicit = output.get("important") is True or isinstance(
                    output.get("decision"), str
                )
                important = (
                    result.pipeline == "fact_claim"
                    or explicit
                    or (
                        classification is not None
                        and classification.confidence
                        >= self.settings.ltm_min_confidence
                        and isinstance(text, str)
                        and len(text) >= self.settings.ltm_min_response_chars
                    )
                )
                if not important:
                    return
                if result.pipeline == "fact_claim":
                    claim, status, sources = evidence_fields(output, turn.raw_text)
                    await self.repository.add_evidence(
                        db, turn, result, claim, status, sources
                    )
                if isinstance(text, str) and text.strip():
                    decision = (
                        output.get("decision")
                        if isinstance(output.get("decision"), str)
                        else "response_generated"
                    )
                    await self.repository.add_decision(db, turn, result, decision, text)
                await self.repository.add_ethics_placeholder(db, turn.id)

    async def _perform(self, job):
        kind, identifier = job
        if kind == "result":
            await self.promote_result(identifier)
            return
        if kind == "end":
            await self.repository.end_session(identifier)
        offset = 0
        while True:
            results = await self.repository.result_page(identifier, offset)
            if not results:
                return
            for result_id in results:
                await self.promote_result(result_id, force=True)
            offset += len(results)

    async def _run(self):
        while True:
            job = await self.queue.get()
            try:
                for attempt in range(self.settings.persistence_retries + 1):
                    try:
                        await asyncio.wait_for(
                            self._perform(job), self.settings.ltm_job_timeout_sec
                        )
                        break
                    except Exception:
                        if attempt == self.settings.persistence_retries:
                            self.failed_jobs += 1
                            logger.exception(
                                "Long-term promotion failed; session replay can recover: %s",
                                job,
                            )
                        else:
                            await asyncio.sleep(
                                self.settings.persistence_retry_sec * 2**attempt
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
                logger.warning("Long-term promotion shutdown timed out")
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
