"""Alembic migrations.

Local sqlite uses metadata.create_all unless RUN_MIGRATIONS=true.
Postgres in docker-compose should run: alembic upgrade head
"""

import asyncio
from logging.config import fileConfig
from typing import Any

from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.repositories.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _is_async_url(url: str) -> bool:
    return "+asyncpg" in url or "+aiosqlite" in url


def _do_run_migrations(connection: Any) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    if _is_async_url(config.get_main_option("sqlalchemy.url") or ""):
        connectable = async_engine_from_config(
            section, prefix="sqlalchemy.", poolclass=pool.NullPool
        )

        async def _run() -> None:
            async with connectable.connect() as connection:
                await connection.run_sync(_do_run_migrations)
            await connectable.dispose()

        asyncio.run(_run())
        return

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
