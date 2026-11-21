"""Run async PostgreSQL migrations using the application's environment settings."""

import asyncio
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from backend.app.core.config import Settings
from backend.app.models import Base

target_metadata = Base.metadata


def migrate(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def online():
    url = context.config.attributes.get("database_url") or Settings().database_url
    engine = create_async_engine(url)
    async with engine.connect() as connection:
        await connection.run_sync(migrate)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url, target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())
