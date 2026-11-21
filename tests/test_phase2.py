"""Phase 2 memory behavior, including real Memcached and PostgreSQL tests."""

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.app.core.config import Settings
from backend.app.memory.short_term import (
    MemcachedShortTermStore,
    ShortTermMemoryService,
    memory_turn,
)
from backend.app.schemas.discussion import ClassifyRequest
from backend.app.services.context import ConversationContextManager
from tests.test_phase1 import storage


class FakeClient:
    """Versioned in-memory Memcached semantics, including expiry and add races."""

    def __init__(self):
        self.values = {}
        self.version = 0
        self.available = True

    def _read(self, key):
        if not self.available:
            raise ConnectionError("Memcached down")
        value = self.values.get(key)
        if value and value[2] <= time.monotonic():
            del self.values[key]
            return None
        return value

    async def get(self, key):
        value = self._read(key)
        return value[0] if value else None

    async def gets(self, key):
        value = self._read(key)
        await asyncio.sleep(0)  # Force concurrent readers to observe the same token.
        return (value[0], value[1]) if value else (None, None)

    async def add(self, key, value, exptime):
        if self._read(key) is not None:
            return False
        return self._write(key, value, exptime)

    async def cas(self, key, value, token, exptime):
        current = self._read(key)
        if current is None or current[1] != token:
            return False
        return self._write(key, value, exptime)

    def _write(self, key, value, exptime):
        self.version += 1
        self.values[key] = (value, self.version, time.monotonic() + exptime)
        return True

    async def close(self):
        pass


def setup_memory(**overrides):
    settings = Settings(
        _env_file=None, memory_timeout_sec=0.3, memory_cas_retries=20, **overrides
    )
    client = FakeClient()
    store = MemcachedShortTermStore(settings, client)
    repository = SimpleNamespace(
        memory_window=AsyncMock(return_value=[]),
        memory_state=AsyncMock(
            return_value={
                "current_topic": None,
                "active_speaker": None,
                "pending_claims": [],
            }
        ),
    )
    return ShortTermMemoryService(store, repository, settings), client


@pytest.mark.asyncio
async def test_hit_and_miss_rebuild_from_repository():
    memory, _ = setup_memory()
    turn = ClassifyRequest(mic_id=1, raw_text="Persisted turn")
    memory.repository.memory_window.return_value = [memory_turn(turn)]
    assert await memory.window(turn.session_id) == [memory_turn(turn)]
    assert await memory.window(turn.session_id) == [memory_turn(turn)]
    assert memory.repository.memory_window.await_count == 1
    assert await memory.state(turn.session_id) == {
        "current_topic": None,
        "active_speaker": None,
        "pending_claims": [],
    }
    assert await memory.state(turn.session_id) == {
        "current_topic": None,
        "active_speaker": None,
        "pending_claims": [],
    }
    assert memory.repository.memory_state.await_count == 1


@pytest.mark.asyncio
async def test_concurrent_cas_adds_do_not_lose_updates_and_retries_deduplicate():
    memory, client = setup_memory(context_window_turns=30)
    session_id = uuid4()
    turns = [
        memory_turn(
            ClassifyRequest(session_id=session_id, mic_id=1, raw_text=f"Turn {i}")
        )
        for i in range(15)
    ]
    await asyncio.gather(
        *(memory.store.merge_window(session_id, [turn]) for turn in turns)
    )
    await memory.store.merge_window(session_id, turns)
    window = await memory.window(session_id)
    assert len(window) == 15
    assert {turn["id"] for turn in window} == {turn["id"] for turn in turns}
    assert list(client.values)[0] == f"stm:{session_id}:window".encode()


@pytest.mark.asyncio
async def test_byte_and_turn_limits():
    memory, _ = setup_memory(context_window_turns=2, memory_max_bytes=1024)
    session_id = uuid4()
    turns = [
        memory_turn(
            ClassifyRequest(session_id=session_id, mic_id=1, raw_text="界" * 200)
        )
        for _ in range(5)
    ]
    result = await memory.store.merge_window(session_id, turns)
    assert len(result) == 1
    assert len(memory.store.encode(result)) <= 1024
    assert result[-1]["id"] == turns[-1]["id"]


@pytest.mark.asyncio
async def test_expiry_rebuild_and_memcached_down():
    memory, client = setup_memory(session_ttl_seconds=1)
    turn = ClassifyRequest(mic_id=1, raw_text="Durable turn")
    memory.repository.memory_window.return_value = [memory_turn(turn)]
    await memory.window(turn.session_id)
    await asyncio.sleep(1.05)
    await memory.window(turn.session_id)
    assert memory.repository.memory_window.await_count == 2
    client.available = False
    assert await memory.window(turn.session_id) == [memory_turn(turn)]
    assert memory.repository.memory_window.await_count == 3


@pytest.mark.asyncio
async def test_corrupt_cache_rebuilds_without_unconditional_overwrite():
    memory, client = setup_memory()
    session_id = uuid4()
    await client.add(memory.store.key(session_id, "window"), b"invalid JSON", 60)
    assert await memory.window(session_id) == []
    assert await memory.store.get_window(session_id) == []


@pytest.mark.asyncio
async def test_state_cas_keeps_newest_speaker_and_topic():
    memory, _ = setup_memory()
    session_id = uuid4()
    new = {
        "current_topic": "Education",
        "active_speaker": 2,
        "pending_claims": ["Claim"],
        "_state_revision": "2026-11-21T12:00:00",
        "_speaker_order": "b",
    }
    old = {
        "current_topic": None,
        "active_speaker": 1,
        "pending_claims": [],
        "_state_revision": "",
        "_speaker_order": "a",
    }
    await memory.store.merge_state(session_id, new)
    await memory.store.merge_state(session_id, old)
    assert await memory.state(session_id) == new


@pytest.mark.asyncio
async def test_background_cache_failure_does_not_undo_postgres(storage):
    memory, client = setup_memory()
    memory.repository = storage.repository
    storage.memory = memory
    client.available = False
    memory.start()
    turn = ClassifyRequest(mic_id=2, raw_text="Must remain durable")
    assert storage.submit(turn=turn)
    await asyncio.wait_for(storage.queue.join(), 5)
    assert storage.failed_jobs == 0
    assert (await storage.repository.utterances(turn.session_id))[
        0
    ].raw_text == turn.raw_text
    await memory.close()


@pytest.mark.asyncio
async def test_real_postgres_rebuild_and_write_after_commit(storage):
    memory, _ = setup_memory()
    memory.repository = storage.repository
    storage.memory = memory
    first = ClassifyRequest(mic_id=1, raw_text="First durable turn")
    second = ClassifyRequest(
        session_id=first.session_id, mic_id=2, raw_text="Second durable turn"
    )
    memory.start()
    try:
        storage.submit(turn=first)
        storage.submit(turn=second)
        await asyncio.wait_for(storage.queue.join(), 5)
        await asyncio.wait_for(memory.queue.join(), 5)
        assert [turn["text"] for turn in await memory.window(first.session_id)] == [
            first.raw_text,
            second.raw_text,
        ]
        state = await memory.state(first.session_id)
        assert state["active_speaker"] == 2
        state = await memory.update_state(
            first.session_id,
            {"current_topic": "Education", "pending_claims": ["A claim"]},
        )
        assert state["current_topic"] == "Education"
        fresh, _ = setup_memory()
        fresh.repository = storage.repository
        assert [t["text"] for t in await fresh.window(first.session_id)] == [
            first.raw_text,
            second.raw_text,
        ]
        assert (await fresh.state(first.session_id))["pending_claims"] == ["A claim"]
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_context_reads_cache_and_excludes_current_turn():
    memory, _ = setup_memory()
    first = ClassifyRequest(mic_id=1, raw_text="Previous")
    current = ClassifyRequest(session_id=first.session_id, mic_id=2, raw_text="Current")
    await memory.store.merge_window(
        first.session_id, [memory_turn(first), memory_turn(current)]
    )
    context = ConversationContextManager(
        SimpleNamespace(submit=lambda **kwargs: True), memory.settings, memory
    )
    assert await context.read(current, []) == [{"mic_id": 1, "text": "Previous"}]


@pytest.mark.asyncio
async def test_context_uses_local_fallback_if_cache_and_postgres_are_down():
    memory, client = setup_memory()
    client.available = False
    memory.repository.memory_window.side_effect = ConnectionError("PostgreSQL down")
    context = ConversationContextManager(
        SimpleNamespace(submit=lambda **kwargs: True), memory.settings, memory
    )
    current = ClassifyRequest(mic_id=1, raw_text="Current")
    fallback = [{"mic_id": 2, "text": "Last locally observed turn"}]
    assert await context.read(current, fallback) == fallback


@pytest.mark.asyncio
async def test_failed_postgres_transaction_never_updates_cache(storage):
    memory, client = setup_memory()
    memory.repository = storage.repository
    storage.memory = memory
    turn = ClassifyRequest(mic_id=1, raw_text="Must not appear in cache")
    from unittest.mock import patch

    with patch.object(
        storage.repository,
        "ensure_turn",
        AsyncMock(side_effect=RuntimeError("rollback")),
    ):
        with pytest.raises(RuntimeError, match="rollback"):
            await storage._write((turn, None, None, None))
    assert memory.queue.empty()
    assert await memory.store.get_window(turn.session_id) is None
    assert await storage.repository.utterances(turn.session_id) == []


@pytest.mark.asyncio
async def test_late_rebuild_merges_rather_than_replaces_new_cached_turns():
    memory, _ = setup_memory()
    session_id = uuid4()
    old = memory_turn(
        ClassifyRequest(session_id=session_id, mic_id=1, raw_text="Older DB snapshot")
    )
    new = memory_turn(
        ClassifyRequest(session_id=session_id, mic_id=2, raw_text="Concurrent commit")
    )

    async def rebuild(*args):
        await memory.store.merge_window(session_id, [new])
        return [old]

    memory.repository.memory_window.side_effect = rebuild
    assert [turn["text"] for turn in await memory.window(session_id)] == [
        old["text"],
        new["text"],
    ]


@pytest.mark.asyncio
async def test_real_memcached_cas_and_expiry():
    port = os.getenv("TEST_MEMCACHED_PORT")
    if not port:
        pytest.skip("Set TEST_MEMCACHED_PORT for isolated Memcached integration")
    settings = Settings(
        _env_file=None,
        memcached_port=int(port),
        session_ttl_seconds=1,
        memory_timeout_sec=1,
        memory_cas_retries=30,
    )
    store = MemcachedShortTermStore(settings)
    session_id = uuid4()
    turns = [
        memory_turn(ClassifyRequest(session_id=session_id, mic_id=1, raw_text=str(i)))
        for i in range(10)
    ]
    try:
        assert await store.get_window(session_id) is None
        await asyncio.gather(
            *(store.merge_window(session_id, [turn]) for turn in turns)
        )
        assert len(await store.get_window(session_id)) == 10
        await asyncio.sleep(2.1)  # Memcached expires on second boundaries.
        assert await store.get_window(session_id) is None
    finally:
        await store.close()
