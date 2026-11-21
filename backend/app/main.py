"""FastAPI application factory and lifecycle management."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from backend.app.api.routes import mics, pages, ws, discussion
from backend.app.core.config import get_settings
from backend.app.core.database import Database
from backend.app.pipelines.manager import PipelineManager
from backend.app.services.transcripts import TranscriptService
from backend.app.repositories.discussion import DiscussionRepository
from backend.app.services.persistence import PersistenceService
from backend.app.services.context import ConversationContextManager
from backend.app.services.classifier import IntentClassifier
from backend.app.services.orchestration import OrchestrationEngine
from backend.app.pipelines.discussion import build_pipelines
from openai import AsyncOpenAI
from backend.app.memory.short_term import (
    MemcachedShortTermStore,
    ShortTermMemoryService,
)
from backend.app.repositories.long_term import LongTermMemoryRepository
from backend.app.services.promotion import MemoryPromotionService


def configure_logging(level: str) -> None:
    """Configure concise application logging once at startup."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create and clean up shared services for the application."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.database = Database(settings)
    app.state.discussion_repository = DiscussionRepository(app.state.database)
    memory = ShortTermMemoryService(
        MemcachedShortTermStore(settings), app.state.discussion_repository, settings
    )
    promotion = MemoryPromotionService(
        LongTermMemoryRepository(app.state.database), settings
    )
    persistence = PersistenceService(
        app.state.discussion_repository, settings, memory, promotion
    )
    client = (
        AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.llm_timeout_sec,
            max_retries=0,
        )
        if settings.groq_api_key
        else None
    )
    orchestration = OrchestrationEngine(
        ConversationContextManager(persistence, settings, memory),
        IntentClassifier(client, settings),
        build_pipelines(client, settings),
        persistence,
        settings,
    )
    app.state.orchestration = orchestration
    app.state.persistence = persistence
    app.state.short_term_memory = memory
    app.state.long_term_memory = promotion
    app.state.pipeline_manager = PipelineManager(
        settings, TranscriptService(persistence), orchestration
    )
    await app.state.database.init_db()
    memory.start()
    promotion.start()
    persistence.start()
    orchestration.start()
    app.state.pipeline_manager.start_workers()
    try:
        yield
    finally:
        await app.state.pipeline_manager.shutdown_workers()
        await orchestration.close()
        await persistence.close()
        promotion.promote_session(orchestration.session_id, end=True)
        await promotion.close()
        await memory.close()
        if client:
            await client.close()
        await app.state.database.close_db()


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""
    app = FastAPI(title="Two Microphone Podcast", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")
    app.include_router(pages.router)
    app.include_router(mics.router)
    app.include_router(ws.router)
    app.include_router(discussion.router)

    @app.get("/health")
    async def health_check(request: Request) -> dict[str, str]:
        """Verify that a database session can execute PostgreSQL work."""
        try:
            await request.app.state.database.health_check()
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="Database unavailable"
            ) from error
        return {"status": "ok", "database": "connected"}

    return app


app = create_app()
