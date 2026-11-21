"""FastAPI application factory and lifecycle management."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from backend.app.api.routes import mics, pages, ws
from backend.app.core.config import get_settings
from backend.app.core.database import Database
from backend.app.pipelines.manager import PipelineManager
from backend.app.services.transcripts import TranscriptService


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
    app.state.pipeline_manager = PipelineManager(
        settings, TranscriptService(app.state.database)
    )
    await app.state.database.init_db()
    app.state.pipeline_manager.start_workers()
    try:
        yield
    finally:
        await app.state.pipeline_manager.shutdown_workers()
        await app.state.database.close_db()


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""
    app = FastAPI(title="Two Microphone Podcast", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")
    app.include_router(pages.router)
    app.include_router(mics.router)
    app.include_router(ws.router)

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
