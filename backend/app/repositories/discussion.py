"""Transactional PostgreSQL repository; no pipeline imports SQLAlchemy."""

from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from backend.app.models import (
    Session,
    Participant,
    Utterance,
    IntentClassification,
    PipelineResult,
)


class DiscussionRepository:
    def __init__(self, database):
        self.database = database

    async def ensure_turn(self, db, turn):
        await db.execute(
            insert(Session)
            .values(id=turn.session_id, started_at=turn.started_at, metadata={})
            .on_conflict_do_nothing()
        )
        await db.execute(
            insert(Participant)
            .values(
                id=uuid4(),
                session_id=turn.session_id,
                mic_id=turn.mic_id,
                display_name=turn.display_name,
            )
            .on_conflict_do_nothing(index_elements=["session_id", "mic_id"])
        )
        participant = await db.scalar(
            select(Participant).where(
                Participant.session_id == turn.session_id,
                Participant.mic_id == turn.mic_id,
            )
        )
        await db.execute(
            insert(Utterance)
            .values(
                id=turn.utterance_id,
                session_id=turn.session_id,
                participant_id=participant.id,
                raw_text=turn.raw_text,
                started_at=turn.started_at,
                ended_at=turn.ended_at,
                stt_model=turn.stt_model,
                stt_confidence=turn.stt_confidence,
            )
            .on_conflict_do_nothing()
        )
        saved = await db.get(Utterance, turn.utterance_id)
        if (
            saved.session_id != turn.session_id
            or saved.raw_text != turn.raw_text
            or saved.participant_id != participant.id
        ):
            raise ValueError("utterance_id already belongs to a different turn")

    async def save_classification(self, db, turn, classification, audit):
        values = classification.model_dump(mode="json")
        row_id = await db.scalar(
            insert(IntentClassification)
            .values(id=uuid4(), utterance_id=turn.utterance_id, **values)
            .on_conflict_do_nothing(index_elements=["utterance_id"])
            .returning(IntentClassification.id)
        )
        if row_id is not None:
            await audit.append(
                db,
                turn.session_id,
                "intent_classified",
                {
                    "classification_id": str(row_id),
                    "utterance_id": str(turn.utterance_id),
                    "session_id": str(turn.session_id),
                    **values,
                },
            )

    async def save_result(self, db, turn, result):
        await db.execute(
            insert(PipelineResult)
            .values(utterance_id=turn.utterance_id, **result)
            .on_conflict_do_nothing(index_elements=["id"])
        )

    async def utterances(self, session_id, limit=100, offset=0):
        async with self.database.session_factory() as db:
            return list(
                (
                    await db.scalars(
                        select(Utterance)
                        .where(Utterance.session_id == session_id)
                        .order_by(Utterance.started_at, Utterance.id)
                        .limit(limit)
                        .offset(offset)
                    )
                ).all()
            )

    async def classification(self, utterance_id):
        async with self.database.session_factory() as db:
            return await db.scalar(
                select(IntentClassification).where(
                    IntentClassification.utterance_id == utterance_id
                )
            )

    async def memory_window(self, session_id, limit):
        async with self.database.session_factory() as db:
            rows = (
                await db.execute(
                    select(Utterance, Participant.mic_id)
                    .join(Participant, Utterance.participant_id == Participant.id)
                    .where(Utterance.session_id == session_id)
                    .order_by(Utterance.started_at.desc(), Utterance.id.desc())
                    .limit(limit)
                )
            ).all()
            return [
                {
                    "id": str(row.id),
                    "mic_id": mic_id,
                    "text": row.raw_text,
                    "started_at": row.started_at.astimezone(timezone.utc).isoformat(),
                }
                for row, mic_id in reversed(rows)
            ]

    async def memory_state(self, session_id):
        async with self.database.session_factory() as db:
            session = await db.get(Session, session_id)
            stored = (session.metadata_ or {}).get("stm_state", {}) if session else {}
        turns = await self.memory_window(session_id, 1)
        latest = turns[-1] if turns else None
        return {
            "current_topic": stored.get("current_topic"),
            "pending_claims": stored.get("pending_claims", []),
            "active_speaker": latest["mic_id"] if latest else None,
            "_speaker_order": (latest["started_at"] + latest["id"]) if latest else "",
            "_state_revision": stored.get("_state_revision", ""),
        }

    async def update_memory_state(self, session_id, changes):
        async with self.database.session_factory() as db:
            async with db.begin():
                session = await db.scalar(
                    select(Session).where(Session.id == session_id).with_for_update()
                )
                if session is None:
                    raise ValueError(
                        "Session must be persisted before state can be updated"
                    )
                metadata = dict(session.metadata_ or {})
                metadata["stm_state"] = {
                    **metadata.get("stm_state", {}),
                    **changes,
                    "_state_revision": datetime.now(timezone.utc).isoformat(),
                }
                session.metadata_ = metadata
        return await self.memory_state(session_id)
