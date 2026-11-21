"""Read-only memory diagnostics; inspecting never repopulates or refreshes TTL."""

import asyncio
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryPart(BaseModel):
    source: Literal["memcached", "postgresql", "unavailable"]
    cache_status: Literal["hit", "miss", "error"]
    data: Any = None
    message: str | None = None


class MemoryInspection(BaseModel):
    session_id: UUID
    window_key: str
    state_key: str
    ttl_seconds: int
    window_limit: int
    max_bytes: int
    pending_cache_updates: int
    pending_database_writes: int
    pending_promotions: int = 0
    window: MemoryPart
    state: MemoryPart
    long_term_status: str = "not_implemented"
    long_term_message: str = (
        "Long-term memory is planned for Phase 3. No evidence, decisions or ethics records are maintained yet."
    )
    long_term_records: list[dict] = Field(default_factory=list)


async def inspect_memory(memory, persistence, session_id, long_term=None):
    async def part(cache_read, database_read):
        try:
            cached = await asyncio.wait_for(
                cache_read(session_id), memory.settings.memory_timeout_sec
            )
            if cached is not None:
                return MemoryPart(source="memcached", cache_status="hit", data=cached)
            status = "miss"
        except Exception:
            status = "error"
        try:
            data = await asyncio.wait_for(
                database_read(), memory.settings.memory_db_timeout_sec
            )
            return MemoryPart(
                source="postgresql",
                cache_status=status,
                data=data,
                message="Read-only PostgreSQL fallback; cache was not changed.",
            )
        except Exception:
            return MemoryPart(
                source="unavailable",
                cache_status=status,
                message="Memory data could not be read. Check PostgreSQL and Memcached.",
            )

    window, state = await asyncio.gather(
        part(
            memory.store.get_window,
            lambda: memory.repository.memory_window(
                session_id, memory.settings.context_window_turns
            ),
        ),
        part(
            memory.store.get_state, lambda: memory.repository.memory_state(session_id)
        ),
    )
    ltm_fields = {}
    if long_term is not None:
        ltm_fields["pending_promotions"] = long_term.queue.qsize()
        try:
            records = await asyncio.wait_for(
                long_term.records(session_id), memory.settings.memory_db_timeout_sec
            )
            ltm_fields.update(
                long_term_status="available",
                long_term_message=f"PostgreSQL long-term memory: {len(records)} records (up to 100 per category). Ethical evaluation is not implemented; placeholder scores remain null.",
                long_term_records=records,
            )
        except Exception:
            ltm_fields.update(
                long_term_status="unavailable",
                long_term_message="Long-term memory could not be read. Apply migrations and check PostgreSQL.",
            )
    return MemoryInspection(
        session_id=session_id,
        window_key=f"stm:{session_id}:window",
        state_key=f"stm:{session_id}:state",
        ttl_seconds=memory.settings.session_ttl_seconds,
        window_limit=memory.settings.context_window_turns,
        max_bytes=memory.settings.memory_max_bytes,
        pending_cache_updates=memory.queue.qsize(),
        pending_database_writes=persistence.queue.qsize(),
        window=window,
        state=state,
        **ltm_fields,
    )
