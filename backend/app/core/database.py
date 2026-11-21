"""Reusable asynchronous PostgreSQL connectivity."""

from collections.abc import AsyncGenerator
from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from backend.app.core.config import Settings


class Database:
    """Own the application's SQLAlchemy engine and asynchronous sessions."""

    def __init__(self, settings: Settings) -> None:
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url, pool_pre_ping=True
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self) -> None:
        """Initialize database resources without creating schema objects."""
        # Schema changes are managed exclusively by `alembic upgrade head`.
        # Connectivity failures are handled by background persistence retries.
        return None

    async def close_db(self) -> None:
        """Dispose all database connections during application shutdown."""
        await self.engine.dispose()

    async def health_check(self) -> None:
        """Raise an exception if PostgreSQL cannot answer a trivial query."""
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional database session for a request."""
    database: Database = request.app.state.database
    session = database.session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
