"""Phase 1 tests. PostgreSQL tests require a dedicated TEST_DATABASE_URL."""

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from backend.app.core.config import Settings
from backend.app.core.database import Database
from backend.app.models import AuditLog, PipelineResult
from backend.app.repositories.discussion import DiscussionRepository
from backend.app.schemas.discussion import ClassifyRequest, Classification, Intent
from backend.app.services.audit import chain_hash
from backend.app.services.classifier import IntentClassifier
from backend.app.services.context import ConversationContextManager
from backend.app.services.orchestration import OrchestrationEngine
from backend.app.services.persistence import PersistenceService
from backend.app.schemas.events import LiveEvent


def prediction():
    return Classification(
        intent=Intent.FACT_CLAIM,
        confidence=0.9,
        model_version="test-model",
        raw_output={"intent": "fact_claim"},
        latency_ms=1,
    )


@pytest_asyncio.fixture
async def storage():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL test database")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("TEST_DATABASE_URL database name must end in _test")
    env = {**os.environ, "DATABASE_URL": url}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env=env,
        check=True,
        capture_output=True,
    )
    settings = Settings(_env_file=None, database_url=url, persistence_retry_sec=0.01)
    database = Database(settings)
    service = PersistenceService(DiscussionRepository(database), settings)
    service.start()
    yield service
    await service.close()
    await database.close_db()


@pytest.mark.asyncio
async def test_postgres_persistence_idempotent_and_audited(storage):
    turn = ClassifyRequest(mic_id=1, raw_text="A verifiable assertion.")
    result = {
        "id": uuid4(),
        "pipeline": "fact_claim",
        "output": {"text": "Result"},
        "status": "completed",
        "latency_ms": 2,
    }
    for _ in range(2):
        assert storage.submit(turn=turn, classification=prediction(), result=result)
    await asyncio.wait_for(storage.queue.join(), 10)
    assert storage.failed_jobs == 0
    turns = await storage.repository.utterances(turn.session_id)
    assert len(turns) == 1 and turns[0].raw_text == turn.raw_text
    saved = await storage.repository.classification(turn.utterance_id)
    assert saved.intent == "fact_claim"
    async with storage.repository.database.session_factory() as db:
        audits = list(
            (
                await db.scalars(
                    select(AuditLog).where(AuditLog.session_id == turn.session_id)
                )
            ).all()
        )
        assert len(audits) == 1
        assert audits[0].hash == chain_hash(audits[0].prev_hash, audits[0].payload)
        results = list(
            (
                await db.scalars(
                    select(PipelineResult).where(
                        PipelineResult.utterance_id == turn.utterance_id
                    )
                )
            ).all()
        )
        assert len(results) == 1


@pytest.mark.asyncio
async def test_postgres_concurrent_audit_chain_and_immutability(storage):
    turns = [ClassifyRequest(mic_id=1, raw_text=f"Claim {i}") for i in range(4)]
    await asyncio.gather(
        *[storage._write((turn, prediction(), None, None)) for turn in turns]
    )
    async with storage.repository.database.session_factory() as db:
        rows = list((await db.scalars(select(AuditLog).order_by(AuditLog.id))).all())
        for previous, current in zip(rows, rows[1:]):
            assert current.prev_hash == previous.hash
        assert all(row.hash == chain_hash(row.prev_hash, row.payload) for row in rows)
        for statement in (
            "UPDATE audit_log SET event_type='tampered'",
            "DELETE FROM audit_log",
            "TRUNCATE audit_log",
        ):
            with pytest.raises(Exception, match="append-only"):
                await db.execute(text(statement))
            await db.rollback()


@pytest.mark.asyncio
async def test_classifier_validates_output():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        id="response",
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content='{"intent":"question_query","confidence":0.8}'
                                )
                            )
                        ],
                    )
                )
            )
        )
    )
    classifier = IntentClassifier(client, Settings(_env_file=None))
    result = await classifier.classify(ClassifyRequest(mic_id=1, raw_text="Why?"), [])
    assert result.intent == Intent.QUESTION_QUERY and result.confidence == 0.8
    client.chat.completions.create.return_value.choices[0].message.content = (
        '{"intent":"invalid","confidence":5}'
    )
    with pytest.raises(ValueError):
        await classifier.classify(ClassifyRequest(mic_id=1, raw_text="Why?"), [])


@pytest.mark.asyncio
async def test_persistence_outage_retries_without_blocking_producer():
    settings = Settings(
        _env_file=None,
        persistence_queue_size=1,
        persistence_retries=1,
        persistence_retry_sec=0.01,
    )
    service = PersistenceService(None, settings)
    service._write = AsyncMock(side_effect=ConnectionError("database down"))
    turn = ClassifyRequest(mic_id=1, raw_text="Hello")
    assert service.submit(turn=turn)
    assert not service.submit(turn=turn)
    service.start()
    await asyncio.wait_for(service.queue.join(), 1)
    assert service._write.await_count == 2
    assert service.failed_jobs == 2
    await service.close()


@pytest.mark.asyncio
async def test_context_and_orchestration_routing():
    settings = Settings(_env_file=None)
    persistence = SimpleNamespace(submit=lambda **kwargs: True)
    context = ConversationContextManager(persistence, settings)
    classifier = SimpleNamespace(classify=AsyncMock(return_value=prediction()))
    engine = OrchestrationEngine(context, classifier, {}, persistence, settings)
    turn = ClassifyRequest(mic_id=1, raw_text="Claim")
    result, window = await engine.classify(turn)
    assert result.persistence == "queued" and window == []
    result, window = await engine.classify(
        ClassifyRequest(session_id=turn.session_id, mic_id=2, raw_text="Next")
    )
    assert window[0]["text"] == "Claim"


@pytest.mark.asyncio
async def test_read_endpoints_and_classify_without_audio_models(storage):
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport
    from backend.app.api.routes.discussion import router

    app = FastAPI()
    app.include_router(router)
    settings = Settings(_env_file=None)
    app.state.discussion_repository = storage.repository
    app.state.orchestration = OrchestrationEngine(
        ConversationContextManager(storage, settings),
        SimpleNamespace(classify=AsyncMock(return_value=prediction())),
        {},
        storage,
        settings,
    )
    turn = ClassifyRequest(mic_id=1, raw_text="A test claim")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/classify", json=turn.model_dump(mode="json"))
        assert response.status_code == 200
        assert response.json()["persistence"] == "queued"
        await asyncio.wait_for(storage.queue.join(), 10)
        response = await client.get(f"/sessions/{turn.session_id}/utterances")
        assert (
            response.status_code == 200
            and response.json()[0]["raw_text"] == turn.raw_text
        )
        response = await client.get(f"/utterances/{turn.utterance_id}/classification")
        assert response.status_code == 200 and response.json()["intent"] == "fact_claim"
        assert (
            await client.get(f"/utterances/{uuid4()}/classification")
        ).status_code == 404
        assert (
            await client.post("/classify", json={"mic_id": 9, "raw_text": ""})
        ).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", list(Intent))
async def test_live_orchestration_routes_one_pipeline_without_blocking_stt(intent):
    settings = Settings(_env_file=None)
    jobs = []
    persistence = SimpleNamespace(submit=lambda **job: jobs.append(job) or True)
    selected = prediction().model_copy(update={"intent": intent})
    classifier = SimpleNamespace(classify=AsyncMock(return_value=selected))
    pipelines = {
        name: SimpleNamespace(run=AsyncMock(return_value={"text": "Agent response"}))
        for name in Intent
    }
    engine = OrchestrationEngine(
        ConversationContextManager(persistence, settings),
        classifier,
        pipelines,
        persistence,
        settings,
    )
    publish = AsyncMock()
    event = LiveEvent(type="transcript.final", mic_id=2, text="A complete utterance")
    engine.enqueue(event, publish)
    assert len(jobs) == 1  # Raw turn saved immediately on context reception.
    assert classifier.classify.await_count == 0  # Audio task never awaits the LLM.
    engine.start()
    await asyncio.wait_for(engine.queue.join(), 2)
    assert pipelines[intent].run.await_count == 1
    assert sum(p.run.await_count for p in pipelines.values()) == 1
    assert len(jobs) == 3
    assert jobs[-1]["result"]["status"] == "completed"
    assert publish.call_args.args[0].source_event_id == event.event_id
    await engine.close()


@pytest.mark.asyncio
async def test_audit_failure_rolls_back_classification(storage):
    turn = ClassifyRequest(mic_id=1, raw_text="Atomic claim")
    audit = storage.audit
    storage.audit = SimpleNamespace(
        append=AsyncMock(side_effect=RuntimeError("audit failed"))
    )
    with pytest.raises(RuntimeError, match="audit failed"):
        await storage._write((turn, prediction(), None, None))
    assert await storage.repository.classification(turn.utterance_id) is None
    storage.audit = audit


@pytest.mark.asyncio
async def test_pipeline_failure_is_persisted_and_transcription_remains_available():
    settings = Settings(_env_file=None)
    jobs = []
    persistence = SimpleNamespace(submit=lambda **job: jobs.append(job) or True)
    engine = OrchestrationEngine(
        ConversationContextManager(persistence, settings),
        SimpleNamespace(classify=AsyncMock(return_value=prediction())),
        {
            Intent.FACT_CLAIM: SimpleNamespace(
                run=AsyncMock(side_effect=RuntimeError("agent down"))
            )
        },
        persistence,
        settings,
    )
    publish = AsyncMock()
    engine.enqueue(
        LiveEvent(type="transcript.final", mic_id=1, text="A claim"), publish
    )
    engine.start()
    await asyncio.wait_for(engine.queue.join(), 2)
    assert jobs[-1]["result"]["status"] == "failed"
    assert publish.call_args.args[0].type == "error"
    await engine.close()
