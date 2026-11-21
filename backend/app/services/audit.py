"""Serialized, transactional append service for the global audit chain."""

import hashlib
import json
from backend.app.repositories.audit import AuditRepository


def canonical_payload(payload: dict) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def chain_hash(prev_hash: str, payload: dict) -> str:
    return hashlib.sha256(
        (prev_hash + canonical_payload(payload)).encode("utf-8")
    ).hexdigest()


class AuditService:
    def __init__(self):
        self.repository = AuditRepository()

    async def append(self, db, session_id, event_type, payload):
        previous = await self.repository.locked_head(db)
        await self.repository.add(
            db, session_id, event_type, payload, previous, chain_hash(previous, payload)
        )
