"""PostgreSQL primitives for serialized audit-chain appends."""

from sqlalchemy import select, func
from backend.app.models import AuditLog


class AuditRepository:
    async def locked_head(self, db) -> str:
        await db.execute(select(func.pg_advisory_xact_lock(437926101)))
        return (
            await db.scalar(select(AuditLog.hash).order_by(AuditLog.id.desc()).limit(1))
            or "0" * 64
        )

    async def add(self, db, session_id, event_type, payload, previous, digest):
        db.add(
            AuditLog(
                session_id=session_id,
                event_type=event_type,
                payload=payload,
                prev_hash=previous,
                hash=digest,
            )
        )
        await db.flush()
