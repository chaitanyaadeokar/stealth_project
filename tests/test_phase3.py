"""Long-term memory promotion against a dedicated PostgreSQL database."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from backend.app.models import Session, LongTermEthics
from backend.app.repositories.long_term import LongTermMemoryRepository
from backend.app.schemas.discussion import ClassifyRequest
from backend.app.services.promotion import MemoryPromotionService, evidence_fields
from backend.app.services.memory_inspection import inspect_memory
from tests.test_phase1 import storage, prediction


def result(pipeline="fact_claim", output=None, status="completed"):
    return {
        "id": uuid4(),
        "pipeline": pipeline,
        "output": output or {"text": "Model assessment only"},
        "status": status,
        "latency_ms": 1,
    }


async def save(storage, turn, output):
    await storage._write((turn, prediction(), output, None))


@pytest.mark.asyncio
async def test_evidence_decision_ethics_idempotent_and_queryable(storage):
    repository = LongTermMemoryRepository(storage.repository.database)
    service = MemoryPromotionService(repository, storage.settings)
    turn = ClassifyRequest(mic_id=1, raw_text="Water is wet")
    completed = result(
        output={
            "text": "Supported evidence",
            "evidence": {
                "claim": turn.raw_text,
                "fact_status": "true",
                "sources": [{"url": "https://example.org/source", "verified": True}],
            },
        }
    )
    await save(storage, turn, completed)
    await asyncio.gather(
        service.promote_result(completed["id"]), service.promote_result(completed["id"])
    )
    evidence, decisions, ethics = await repository.by_session(turn.session_id)
    assert len(evidence) == len(decisions) == len(ethics) == 1
    assert evidence[0].fact_status == "true"
    assert evidence[0].source_result_id == completed["id"]
    assert ethics[0].ethical_score is None and ethics[0].dimensions is None
    assert len(await repository.evidence(claim_text="water", fact_status="true")) >= 1
    assert (
        len(await repository.evidence(session_id=turn.session_id, fact_status="false"))
        == 0
    )
    async with repository.database.session_factory() as db:
        assert await db.scalar(
            text("SELECT dimensions IS NULL FROM ltm_ethics WHERE utterance_id=:id"),
            {"id": turn.utterance_id},
        )


def test_model_prose_is_not_treated_as_verified_evidence():
    claim, status, sources = evidence_fields(
        {"text": "TRUE, according to experts"}, "Claim"
    )
    assert (claim, status, sources) == ("Claim", "uncertain", [])
    _, status, _ = evidence_fields(
        {
            "evidence": {
                "fact_status": "false",
                "sources": [{"url": "https://example.org", "verified": False}],
            }
        },
        "Claim",
    )
    assert status == "uncertain"
    _, status, sources = evidence_fields(
        {
            "evidence": {
                "fact_status": "true",
                "sources": [{"url": "https://[invalid", "verified": True}],
            }
        },
        "Claim",
    )
    assert status == "uncertain" and sources == []


@pytest.mark.asyncio
async def test_low_importance_failed_and_explicit_decisions(storage):
    repository = LongTermMemoryRepository(storage.repository.database)
    service = MemoryPromotionService(repository, storage.settings)
    turn = ClassifyRequest(mic_id=2, raw_text="Hello")
    greeting = result("opinion_debate", {"text": "Hello!"})
    failed = result(status="failed")
    explicit = result(
        "opinion_debate",
        {"text": "Consider both positions.", "decision": "request_more_evidence"},
    )
    for completed in (greeting, failed, explicit):
        await save(storage, turn, completed)
        await service.promote_result(completed["id"])
    evidence, decisions, ethics = await repository.by_session(turn.session_id)
    assert evidence == [] and len(decisions) == 1 and len(ethics) == 1
    assert decisions[0].decision == "request_more_evidence"


@pytest.mark.asyncio
async def test_postcommit_hook_and_session_end_catch_up_late_results(storage):
    settings = storage.settings.model_copy(update={"ltm_promote_on_result": False})
    repository = LongTermMemoryRepository(storage.repository.database)
    service = MemoryPromotionService(repository, settings)
    storage.promotion = service
    service.start()
    turn = ClassifyRequest(mic_id=1, raw_text="Retain this claim")
    first = result()
    try:
        storage.submit(turn=turn, result=first)
        await asyncio.wait_for(storage.queue.join(), 5)
        await asyncio.wait_for(service.queue.join(), 5)
        assert await repository.evidence(session_id=turn.session_id) == []
        assert service.promote_session(turn.session_id, end=True)
        await asyncio.wait_for(service.queue.join(), 5)
        assert len(await repository.evidence(session_id=turn.session_id)) == 1
        async with repository.database.session_factory() as db:
            assert (await db.get(Session, turn.session_id)).ended_at is not None
        late = result()
        storage.submit(turn=turn, result=late)
        await asyncio.wait_for(storage.queue.join(), 5)
        await asyncio.wait_for(service.queue.join(), 5)
        assert len(await repository.evidence(session_id=turn.session_id)) == 2
        service.promote_session(turn.session_id)
        await asyncio.wait_for(service.queue.join(), 5)
        assert len(await repository.evidence(session_id=turn.session_id)) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_failed_commit_does_not_enqueue_promotion(storage):
    service = MemoryPromotionService(
        LongTermMemoryRepository(storage.repository.database), storage.settings
    )
    storage.promotion = service
    turn = ClassifyRequest(mic_id=1, raw_text="Uncommitted")
    with patch.object(
        storage.repository,
        "save_result",
        AsyncMock(side_effect=RuntimeError("rollback")),
    ):
        with pytest.raises(RuntimeError):
            await storage._write((turn, None, result(), None))
    assert service.queue.empty()


@pytest.mark.asyncio
async def test_promotion_outage_retries_and_can_be_replayed():
    from backend.app.core.config import Settings

    service = MemoryPromotionService(
        None,
        Settings(_env_file=None, persistence_retries=1, persistence_retry_sec=0.01),
    )
    service._perform = AsyncMock(side_effect=ConnectionError("db down"))
    assert service.promote_session(uuid4())
    service.start()
    await asyncio.wait_for(service.queue.join(), 1)
    assert service.failed_jobs == 1 and service._perform.await_count == 2
    await service.close()


@pytest.mark.asyncio
async def test_memory_inspector_shows_promoted_records(storage):
    repository = LongTermMemoryRepository(storage.repository.database)
    service = MemoryPromotionService(repository, storage.settings)
    turn = ClassifyRequest(mic_id=1, raw_text="Durable claim")
    completed = result()
    await save(storage, turn, completed)
    await service.promote_result(completed["id"])
    memory = SimpleNamespace(
        settings=storage.settings,
        queue=asyncio.Queue(),
        store=SimpleNamespace(
            get_window=AsyncMock(return_value=[]), get_state=AsyncMock(return_value={})
        ),
    )
    inspection = await inspect_memory(memory, storage, turn.session_id, service)
    assert inspection.long_term_status == "available"
    assert {record["kind"] for record in inspection.long_term_records} == {
        "evidence",
        "decision",
        "ethics",
    }
    assert inspection.model_dump(mode="json")["long_term_records"]


@pytest.mark.asyncio
async def test_session_promotion_end_endpoints(storage):
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport
    from backend.app.api.routes.discussion import router

    service = MemoryPromotionService(
        LongTermMemoryRepository(storage.repository.database), storage.settings
    )
    turn = ClassifyRequest(mic_id=1, raw_text="Session claim")
    await save(storage, turn, result())
    app = FastAPI()
    app.include_router(router)
    app.state.long_term_memory = service
    app.state.orchestration = SimpleNamespace(session_id=turn.session_id)
    service.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post(f"/sessions/{uuid4()}/end")).status_code == 404
            assert (
                await client.post(f"/sessions/{turn.session_id}/memory/promote")
            ).status_code == 202
            ended = await client.post(f"/sessions/{turn.session_id}/end")
            assert ended.status_code == 202
            assert ended.json()["live_session_id"] != str(turn.session_id)
        await asyncio.wait_for(service.queue.join(), 5)
        assert len(await service.repository.evidence(session_id=turn.session_id)) == 1
    finally:
        await service.close()
