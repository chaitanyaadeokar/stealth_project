"""Phase 1 classification and persisted conversation endpoints."""

from uuid import UUID
from fastapi import APIRouter, Request, HTTPException, Query
from backend.app.schemas.discussion import (
    ClassifyRequest,
    ClassifyResponse,
    UtteranceResponse,
    ClassificationResponse,
)
from backend.app.services.memory_inspection import MemoryInspection, inspect_memory

router = APIRouter(tags=["discussion"])


@router.get("/api/memory", response_model=MemoryInspection)
async def memory_inspection(request: Request, session_id: UUID | None = None):
    """Inspect a selected session, defaulting to the current live discussion."""
    response = await inspect_memory(
        request.app.state.short_term_memory,
        request.app.state.persistence,
        session_id or request.app.state.orchestration.session_id,
        getattr(request.app.state, "long_term_memory", None),
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(
        response.model_dump(mode="json"), headers={"Cache-Control": "no-store"}
    )


async def queue_session_promotion(session_id: UUID, request: Request, end=False):
    import asyncio

    service = request.app.state.long_term_memory
    try:
        exists = await asyncio.wait_for(
            service.repository.session_exists(session_id),
            service.settings.memory_db_timeout_sec,
        )
    except Exception:
        raise HTTPException(503, "Long-term storage unavailable")
    if not exists:
        raise HTTPException(404, "Session not found or not persisted yet")
    if not service.promote_session(session_id, end=end):
        raise HTTPException(503, "Promotion queue full; retry later")
    return {"session_id": str(session_id), "promotion": "queued"}


@router.post("/sessions/{session_id}/memory/promote", status_code=202)
async def promote_memory(session_id: UUID, request: Request):
    return await queue_session_promotion(session_id, request)


@router.post("/sessions/{session_id}/end", status_code=202)
async def end_session(session_id: UUID, request: Request):
    response = await queue_session_promotion(session_id, request, end=True)
    if request.app.state.orchestration.session_id == session_id:
        from uuid import uuid4

        request.app.state.orchestration.session_id = uuid4()
    response["session_end"] = "queued"
    response["live_session_id"] = str(request.app.state.orchestration.session_id)
    return response


@router.post("/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest, request: Request):
    try:
        result, _ = await request.app.state.orchestration.classify(body)
        return result
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Classification request failed")
        raise HTTPException(503, "Intent classifier unavailable")


@router.get("/sessions/{session_id}/utterances", response_model=list[UtteranceResponse])
async def utterances(
    session_id: UUID,
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    try:
        return await request.app.state.discussion_repository.utterances(
            session_id, limit, offset
        )
    except Exception:
        raise HTTPException(503, "Conversation storage unavailable")


@router.get(
    "/utterances/{utterance_id}/classification", response_model=ClassificationResponse
)
async def classification(utterance_id: UUID, request: Request):
    try:
        result = await request.app.state.discussion_repository.classification(
            utterance_id
        )
    except Exception:
        raise HTTPException(503, "Conversation storage unavailable")
    if result is None:
        raise HTTPException(
            404, "Classification not persisted yet or utterance not found"
        )
    return result
