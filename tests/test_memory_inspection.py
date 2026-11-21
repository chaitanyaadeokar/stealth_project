"""Memory debug endpoint must report sources without changing the cache."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from backend.app.core.config import Settings
from backend.app.services.memory_inspection import inspect_memory
from backend.app.api.routes.discussion import router


def setup():
    state = {"current_topic": None, "active_speaker": 1, "pending_claims": []}
    store = SimpleNamespace(
        get_window=AsyncMock(return_value=[]),
        get_state=AsyncMock(return_value=state),
        merge_window=AsyncMock(),
        merge_state=AsyncMock(),
    )
    repository = SimpleNamespace(
        memory_window=AsyncMock(return_value=[{"text": "Durable turn"}]),
        memory_state=AsyncMock(return_value=state),
    )
    memory = SimpleNamespace(
        store=store,
        repository=repository,
        settings=Settings(_env_file=None),
        queue=asyncio.Queue(),
    )
    persistence = SimpleNamespace(queue=asyncio.Queue())
    return memory, persistence


@pytest.mark.asyncio
async def test_cache_hit_is_not_repopulated_or_extended():
    memory, persistence = setup()
    result = await inspect_memory(memory, persistence, uuid4())
    assert result.window.source == "memcached"
    assert result.window.cache_status == "hit"
    assert result.window.data == []
    assert result.long_term_status == "not_implemented"
    memory.repository.memory_window.assert_not_awaited()
    memory.repository.memory_state.assert_not_awaited()
    memory.store.merge_window.assert_not_awaited()
    memory.store.merge_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_miss_and_cache_outage_report_postgres_fallback():
    memory, persistence = setup()
    memory.store.get_window.return_value = None
    memory.store.get_state.side_effect = ConnectionError("cache down")
    result = await inspect_memory(memory, persistence, uuid4())
    assert result.window.source == "postgresql"
    assert result.window.cache_status == "miss"
    assert result.state.source == "postgresql"
    assert result.state.cache_status == "error"
    assert result.window.data == [{"text": "Durable turn"}]
    memory.store.merge_window.assert_not_awaited()


@pytest.mark.asyncio
async def test_both_systems_down_are_not_reported_as_empty_memory():
    memory, persistence = setup()
    memory.store.get_window.side_effect = ConnectionError()
    memory.store.get_state.side_effect = ConnectionError()
    memory.repository.memory_window.side_effect = ConnectionError()
    memory.repository.memory_state.side_effect = ConnectionError()
    result = await inspect_memory(memory, persistence, uuid4())
    assert result.window.source == "unavailable"
    assert result.window.data is None
    assert result.state.source == "unavailable"


@pytest.mark.asyncio
async def test_inspection_endpoint_uses_live_session_or_selected_uuid():
    app = FastAPI()
    app.include_router(router)
    memory, persistence = setup()
    app.state.short_term_memory = memory
    app.state.persistence = persistence
    session_id = uuid4()
    app.state.orchestration = SimpleNamespace(session_id=session_id)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/memory")
        assert response.status_code == 200
        assert response.json()["session_id"] == str(session_id)
        assert response.headers["cache-control"] == "no-store"
        chosen = uuid4()
        response = await client.get(f"/api/memory?session_id={chosen}")
        assert response.json()["session_id"] == str(chosen)
        assert (await client.get("/api/memory?session_id=invalid")).status_code == 422
