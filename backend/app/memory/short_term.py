"""CAS-protected Memcached blobs and PostgreSQL-backed short-term memory."""

import asyncio
import json
import logging
from datetime import timezone
from typing import Protocol
from uuid import UUID

import aiomcache

logger = logging.getLogger(__name__)


def memory_turn(turn):
    return {
        "id": str(turn.utterance_id),
        "mic_id": turn.mic_id,
        "text": turn.raw_text,
        "started_at": turn.started_at.astimezone(timezone.utc).isoformat(),
    }


def turn_order(turn):
    return turn["started_at"], turn["id"]


class ShortTermStore(Protocol):
    async def get_window(self, session_id: UUID) -> list[dict] | None: ...
    async def merge_window(self, session_id: UUID, turns: list[dict]) -> list[dict]: ...
    async def get_state(self, session_id: UUID) -> dict | None: ...
    async def merge_state(self, session_id: UUID, state: dict) -> dict: ...
    async def close(self) -> None: ...


class MemcachedShortTermStore:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client or aiomcache.Client(
            settings.memcached_host, settings.memcached_port
        )

    @staticmethod
    def key(session_id, kind):
        return f"stm:{UUID(str(session_id))}:{kind}".encode("ascii")

    def encode(self, value):
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def decode(self, raw, kind):
        value = json.loads(raw)
        if kind == "window":
            if not isinstance(value, list) or any(
                not isinstance(turn, dict)
                or not all(
                    isinstance(turn.get(k), str) for k in ("id", "text", "started_at")
                )
                or turn.get("mic_id") not in (1, 2)
                for turn in value
            ):
                raise ValueError("Invalid cached window")
        elif not isinstance(value, dict) or not all(
            k in value for k in ("current_topic", "active_speaker", "pending_claims")
        ):
            raise ValueError("Invalid cached state")
        return value

    async def _get(self, session_id, kind):
        raw = await asyncio.wait_for(
            self.client.get(self.key(session_id, kind)),
            self.settings.memory_timeout_sec,
        )
        return self.decode(raw, kind) if raw is not None else None

    async def get_window(self, session_id):
        return await self._get(session_id, "window")

    async def get_state(self, session_id):
        return await self._get(session_id, "state")

    def _cap_window(self, turns):
        turns = sorted({turn["id"]: turn for turn in turns}.values(), key=turn_order)
        turns = turns[-self.settings.context_window_turns :]
        while turns and len(self.encode(turns)) > self.settings.memory_max_bytes:
            turns.pop(0)
        return turns

    async def _mutate(self, session_id, kind, merge):
        key = self.key(session_id, kind)
        for attempt in range(self.settings.memory_cas_retries):
            raw, token = await asyncio.wait_for(
                self.client.gets(key), self.settings.memory_timeout_sec
            )
            try:
                current = self.decode(raw, kind) if raw is not None else None
            except (ValueError, TypeError, KeyError):
                current = None
            value = merge(current)
            encoded = self.encode(value)
            if len(encoded) > self.settings.memory_max_bytes:
                raise ValueError("Short-term memory blob exceeds configured byte limit")
            if token is None:
                saved = await asyncio.wait_for(
                    self.client.add(
                        key, encoded, exptime=self.settings.session_ttl_seconds
                    ),
                    self.settings.memory_timeout_sec,
                )
            else:
                saved = await asyncio.wait_for(
                    self.client.cas(
                        key, encoded, token, exptime=self.settings.session_ttl_seconds
                    ),
                    self.settings.memory_timeout_sec,
                )
            if saved:
                return value
            await asyncio.sleep(min(0.001 * 2**attempt, 0.02))
        raise RuntimeError("Short-term memory CAS contention exhausted retries")

    async def merge_window(self, session_id, turns):
        return await self._mutate(
            session_id,
            "window",
            lambda current: self._cap_window([*(current or []), *turns]),
        )

    async def merge_state(self, session_id, state):
        def merge(current):
            current = current or {}
            state_update = dict(state)
            # An older committed turn must not replace a newer active speaker.
            if current.get("_speaker_order", "") > state.get("_speaker_order", ""):
                for key in ("active_speaker", "_speaker_order"):
                    state_update.pop(key, None)
            if current.get("_state_revision", "") > state.get("_state_revision", ""):
                for key in ("current_topic", "pending_claims", "_state_revision"):
                    state_update.pop(key, None)
            return {**current, **state_update}

        return await self._mutate(session_id, "state", merge)

    async def close(self):
        await self.client.close()


class ShortTermMemoryService:
    """Read-through cache; only committed turns enter its background write queue."""

    def __init__(self, store: ShortTermStore, repository, settings):
        self.store = store
        self.repository = repository
        self.settings = settings
        self.queue = asyncio.Queue(maxsize=settings.memory_queue_size)
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self._run(), name="short-term-cache-writer")

    def committed(self, turn):
        try:
            self.queue.put_nowait(turn)
        except asyncio.QueueFull:
            logger.warning(
                "Short-term cache queue full; PostgreSQL will rebuild session %s",
                turn.session_id,
            )

    async def window(self, session_id):
        try:
            cached = await self.store.get_window(session_id)
            if cached is not None:
                return cached
        except Exception:
            logger.warning(
                "Short-term window read failed; using PostgreSQL", exc_info=True
            )
        turns = await asyncio.wait_for(
            self.repository.memory_window(
                session_id, self.settings.context_window_turns
            ),
            self.settings.memory_db_timeout_sec,
        )
        try:
            # CAS merge prevents a delayed rebuild from replacing newer writes.
            return await asyncio.wait_for(
                self.store.merge_window(session_id, turns),
                self.settings.memory_timeout_sec,
            )
        except Exception:
            logger.warning(
                "Short-term window rebuild could not populate cache", exc_info=True
            )
            return turns

    async def state(self, session_id):
        try:
            cached = await self.store.get_state(session_id)
            if cached is not None:
                return cached
        except Exception:
            logger.warning(
                "Short-term state read failed; using PostgreSQL", exc_info=True
            )
        state = await asyncio.wait_for(
            self.repository.memory_state(session_id),
            self.settings.memory_db_timeout_sec,
        )
        try:
            return await asyncio.wait_for(
                self.store.merge_state(session_id, state),
                self.settings.memory_timeout_sec,
            )
        except Exception:
            logger.warning(
                "Short-term state rebuild could not populate cache", exc_info=True
            )
            return state

    async def update_state(self, session_id, changes):
        """Commit explicit topic/claim changes before updating their cache."""
        if set(changes) - {"current_topic", "pending_claims"}:
            raise ValueError("Only current_topic and pending_claims may be updated")
        if (
            "current_topic" in changes
            and changes["current_topic"] is not None
            and not isinstance(changes["current_topic"], str)
        ):
            raise ValueError("current_topic must be a string or null")
        if "pending_claims" in changes and (
            not isinstance(changes["pending_claims"], list)
            or any(not isinstance(claim, str) for claim in changes["pending_claims"])
        ):
            raise ValueError("pending_claims must be a list of strings")
        if len(json.dumps(changes).encode()) > self.settings.memory_max_bytes:
            raise ValueError("State too large")
        state = await asyncio.wait_for(
            self.repository.update_memory_state(session_id, changes),
            self.settings.memory_db_timeout_sec,
        )
        try:
            return await asyncio.wait_for(
                self.store.merge_state(session_id, state),
                self.settings.memory_timeout_sec,
            )
        except Exception:
            logger.warning("Committed state could not populate cache", exc_info=True)
            return state

    async def _run(self):
        while True:
            turn = await self.queue.get()
            try:
                for attempt in range(self.settings.persistence_retries + 1):
                    try:
                        item = memory_turn(turn)
                        # A cold cache must include earlier committed turns too.
                        if await self.store.get_window(turn.session_id) is None:
                            turns = await asyncio.wait_for(
                                self.repository.memory_window(
                                    turn.session_id, self.settings.context_window_turns
                                ),
                                self.settings.memory_db_timeout_sec,
                            )
                            await self.store.merge_window(turn.session_id, turns)
                        await self.store.merge_window(turn.session_id, [item])
                        # Topic and pending claims are durable session metadata.
                        state = await asyncio.wait_for(
                            self.repository.memory_state(turn.session_id),
                            self.settings.memory_db_timeout_sec,
                        )
                        await self.store.merge_state(turn.session_id, state)
                        break
                    except Exception:
                        if attempt == self.settings.persistence_retries:
                            logger.warning(
                                "Short-term cache update exhausted retries for %s",
                                turn.session_id,
                                exc_info=True,
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
                logger.warning("Short-term cache shutdown timed out")
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.store.close()
