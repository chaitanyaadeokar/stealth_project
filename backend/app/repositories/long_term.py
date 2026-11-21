"""Durable promotion inputs and long-term memory queries."""

from uuid import uuid4
from sqlalchemy import select, update, func
from sqlalchemy.dialects.postgresql import insert
from backend.app.models import (
    Session,
    Utterance,
    IntentClassification,
    PipelineResult,
    LongTermEvidence,
    LongTermDecision,
    LongTermEthics,
)


class LongTermMemoryRepository:
    def __init__(self, database):
        self.database = database

    async def source(self, db, result_id):
        return (
            await db.execute(
                select(
                    PipelineResult, Utterance, IntentClassification, Session.ended_at
                )
                .join(Utterance, PipelineResult.utterance_id == Utterance.id)
                .join(Session, Utterance.session_id == Session.id)
                .outerjoin(
                    IntentClassification,
                    IntentClassification.utterance_id == Utterance.id,
                )
                .where(PipelineResult.id == result_id)
            )
        ).first()

    async def add_evidence(self, db, turn, result, claim, fact_status, sources):
        await db.execute(
            insert(LongTermEvidence)
            .values(
                id=uuid4(),
                session_id=turn.session_id,
                utterance_id=turn.id,
                source_result_id=result.id,
                claim=claim,
                fact_status=fact_status,
                sources=sources,
            )
            .on_conflict_do_nothing(index_elements=["source_result_id"])
        )

    async def add_decision(self, db, turn, result, decision, rationale):
        await db.execute(
            insert(LongTermDecision)
            .values(
                id=uuid4(),
                session_id=turn.session_id,
                utterance_id=turn.id,
                source_result_id=result.id,
                agent=result.pipeline,
                decision=decision,
                rationale=rationale,
            )
            .on_conflict_do_nothing(index_elements=["source_result_id"])
        )

    async def add_ethics_placeholder(self, db, utterance_id):
        await db.execute(
            insert(LongTermEthics)
            .values(
                id=uuid4(),
                utterance_id=utterance_id,
                ethical_score=None,
                dimensions=None,
            )
            .on_conflict_do_nothing(index_elements=["utterance_id"])
        )

    async def session_exists(self, session_id):
        async with self.database.session_factory() as db:
            return await db.get(Session, session_id) is not None

    async def end_session(self, session_id):
        async with self.database.session_factory() as db:
            async with db.begin():
                await db.execute(
                    update(Session)
                    .where(Session.id == session_id, Session.ended_at.is_(None))
                    .values(ended_at=func.now())
                )

    async def result_page(self, session_id, offset, limit=100):
        async with self.database.session_factory() as db:
            return list(
                (
                    await db.scalars(
                        select(PipelineResult.id)
                        .join(Utterance, PipelineResult.utterance_id == Utterance.id)
                        .where(Utterance.session_id == session_id)
                        .order_by(PipelineResult.created_at, PipelineResult.id)
                        .offset(offset)
                        .limit(limit)
                    )
                ).all()
            )

    async def evidence(
        self, session_id=None, claim_text=None, fact_status=None, limit=100, offset=0
    ):
        query = select(LongTermEvidence)
        if session_id is not None:
            query = query.where(LongTermEvidence.session_id == session_id)
        if claim_text:
            query = query.where(
                LongTermEvidence.claim.icontains(claim_text, autoescape=True)
            )
        if fact_status:
            query = query.where(LongTermEvidence.fact_status == fact_status)
        async with self.database.session_factory() as db:
            return list(
                (
                    await db.scalars(
                        query.order_by(LongTermEvidence.created_at, LongTermEvidence.id)
                        .limit(limit)
                        .offset(offset)
                    )
                ).all()
            )

    async def by_session(self, session_id, limit=100):
        async with self.database.session_factory() as db:
            evidence = await self.evidence(session_id=session_id, limit=limit)
            decisions = list(
                (
                    await db.scalars(
                        select(LongTermDecision)
                        .where(LongTermDecision.session_id == session_id)
                        .order_by(LongTermDecision.created_at, LongTermDecision.id)
                        .limit(limit)
                    )
                ).all()
            )
            ethics = list(
                (
                    await db.scalars(
                        select(LongTermEthics)
                        .join(Utterance, LongTermEthics.utterance_id == Utterance.id)
                        .where(Utterance.session_id == session_id)
                        .order_by(LongTermEthics.created_at, LongTermEthics.id)
                        .limit(limit)
                    )
                ).all()
            )
            return evidence, decisions, ethics
